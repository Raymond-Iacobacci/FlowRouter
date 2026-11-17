//! Order-preserving Dinic max-flow kernel used by the Python compiler.
//!
//! Python still owns graph construction and the colour/legalisation pass.  This
//! module only replaces the allocation-heavy inner max-flow loop.  Node and edge
//! order are supplied by Python, and ties are broken by the original edge index,
//! so the selected augmenting paths remain identical to the reference code.

use std::collections::{HashMap, VecDeque};
use std::slice;

#[derive(Clone, Copy)]
struct Edge {
    v: usize,
    flow: i64,
    capacity: i64,
    rev: usize,
}

struct Dinic<'a> {
    source: usize,
    sink: usize,
    adjacency: Vec<Vec<usize>>,
    edges: Vec<Edge>,
    level: Vec<i32>,
    blocked: &'a [u8],
}

impl Dinic<'_> {
    #[inline]
    fn usable(&self, edge_index: usize) -> bool {
        let edge = self.edges[edge_index];
        edge.flow < edge.capacity && self.blocked[edge.v] == 0
    }

    fn build_levels(&mut self) -> bool {
        self.level.fill(-1);
        self.level[self.source] = 0;
        let mut queue = VecDeque::with_capacity(self.level.len());
        queue.push_back(self.source);

        while let Some(u) = queue.pop_front() {
            for position in 0..self.adjacency[u].len() {
                let edge_index = self.adjacency[u][position];
                let v = self.edges[edge_index].v;
                if self.level[v] < 0 && self.usable(edge_index) {
                    self.level[v] = self.level[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        self.level[self.sink] >= 0
    }

    fn send_one(&mut self, u: usize, start: &mut [usize]) -> bool {
        if u == self.sink {
            return true;
        }

        while start[u] < self.adjacency[u].len() {
            let edge_index = self.adjacency[u][start[u]];
            let edge = self.edges[edge_index];
            let v = edge.v;
            if self.level[v] == self.level[u] + 1
                && edge.flow < edge.capacity
                && self.blocked[v] == 0
                && self.send_one(v, start)
            {
                let reverse = self.edges[edge_index].rev;
                self.edges[edge_index].flow += 1;
                self.edges[reverse].flow -= 1;
                return true;
            }
            start[u] += 1;
        }
        false
    }

    fn max_flow(&mut self) -> usize {
        let mut augmentations = 0;
        while self.build_levels() {
            let mut start = vec![0; self.adjacency.len()];
            while self.send_one(self.source, &mut start) {
                augmentations += 1;
            }
        }
        augmentations
    }
}

/// Runs Dinic in place and writes the final per-edge flow back to `flows`.
///
/// `ranks` contains five i64 values per node.  Python's stable adjacency sort is
/// reproduced by comparing that tuple and then the original edge index.
/// Returning `usize::MAX` signals invalid input; the Python caller then uses its
/// reference implementation.
#[no_mangle]
pub unsafe extern "C" fn topols_dinic_run(
    node_count: usize,
    edge_count: usize,
    source: usize,
    sink: usize,
    edge_sources: *const u32,
    edge_targets: *const u32,
    capacities: *const i64,
    flows: *mut i64,
    blocked: *const u8,
    ranks: *const i64,
) -> usize {
    if node_count == 0
        || source >= node_count
        || sink >= node_count
        || edge_sources.is_null()
        || edge_targets.is_null()
        || capacities.is_null()
        || flows.is_null()
        || blocked.is_null()
        || ranks.is_null()
    {
        return usize::MAX;
    }

    let sources = slice::from_raw_parts(edge_sources, edge_count);
    let targets = slice::from_raw_parts(edge_targets, edge_count);
    let capacities = slice::from_raw_parts(capacities, edge_count);
    let flow_values = slice::from_raw_parts_mut(flows, edge_count);
    let blocked = slice::from_raw_parts(blocked, node_count);
    let ranks = slice::from_raw_parts(ranks, node_count * 5);

    let mut reverse_lookup = HashMap::with_capacity(edge_count);
    let mut adjacency = vec![Vec::new(); node_count];
    for edge_index in 0..edge_count {
        let u = sources[edge_index] as usize;
        let v = targets[edge_index] as usize;
        if u >= node_count || v >= node_count {
            return usize::MAX;
        }
        reverse_lookup.insert((u, v), edge_index);
        adjacency[u].push(edge_index);
    }

    let mut edges = Vec::with_capacity(edge_count);
    for edge_index in 0..edge_count {
        let u = sources[edge_index] as usize;
        let v = targets[edge_index] as usize;
        let Some(&reverse) = reverse_lookup.get(&(v, u)) else {
            return usize::MAX;
        };
        edges.push(Edge {
            v,
            flow: flow_values[edge_index],
            capacity: capacities[edge_index],
            rev: reverse,
        });
    }

    for edge_indices in &mut adjacency {
        edge_indices.sort_by(|left, right| {
            let left_rank = &ranks[edges[*left].v * 5..edges[*left].v * 5 + 5];
            let right_rank = &ranks[edges[*right].v * 5..edges[*right].v * 5 + 5];
            left_rank.cmp(right_rank).then_with(|| left.cmp(right))
        });
    }

    let mut dinic = Dinic {
        source,
        sink,
        adjacency,
        edges,
        level: vec![-1; node_count],
        blocked,
    };
    let augmentations = dinic.max_flow();
    for (output, edge) in flow_values.iter_mut().zip(dinic.edges.iter()) {
        *output = edge.flow;
    }
    augmentations
}

#[cfg(test)]
mod tests {
    use super::topols_dinic_run;

    #[test]
    fn packs_two_ordered_disjoint_paths() {
        let sources = [0, 1, 1, 3, 0, 2, 2, 3];
        let targets = [1, 0, 3, 1, 2, 0, 3, 2];
        let capacities = [1, 0, 1, 0, 1, 0, 1, 0];
        let mut flows = [0_i64; 8];
        let blocked = [0_u8; 4];
        let ranks = [0_i64; 20];

        let count = unsafe {
            topols_dinic_run(
                4,
                sources.len(),
                0,
                3,
                sources.as_ptr(),
                targets.as_ptr(),
                capacities.as_ptr(),
                flows.as_mut_ptr(),
                blocked.as_ptr(),
                ranks.as_ptr(),
            )
        };

        assert_eq!(count, 2);
        assert_eq!(flows, [1, -1, 1, -1, 1, -1, 1, -1]);
    }
}
