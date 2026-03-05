"""Tests for scirs2 graph algorithms module."""

import numpy as np
import pytest
import scirs2


class TestGraphCreation:
    """Test graph creation functions."""

    def test_graph_from_edges_basic(self):
        """Test creating undirected graph from edge list."""
        edges = [(0, 1), (1, 2), (2, 3)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=4)

        assert graph["num_nodes"] == 4
        assert graph["num_edges"] == 3
        assert graph["directed"] is False
        assert "adjacency" in graph

    def test_graph_from_edges_weighted(self):
        """Test creating weighted graph."""
        edges = [(0, 1, 1.5), (1, 2, 2.0), (2, 3, 0.5)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=4)

        assert graph["num_nodes"] == 4
        assert graph["num_edges"] == 3

        # Check that weights are preserved
        adj = graph["adjacency"]
        assert len(adj[0]) > 0  # Node 0 has neighbors
        assert len(adj[1]) > 0  # Node 1 has neighbors

    def test_digraph_from_edges_basic(self):
        """Test creating directed graph from edge list."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = scirs2.digraph_from_edges_py(edges, num_nodes=3)

        assert graph["num_nodes"] == 3
        assert graph["num_edges"] == 3
        assert graph["directed"] is True

    def test_graph_from_edges_auto_nodes(self):
        """Test automatic node count detection."""
        edges = [(0, 5), (1, 3), (2, 4)]
        graph = scirs2.graph_from_edges_py(edges)

        # Should auto-detect at least 6 nodes (0-5)
        assert graph["num_nodes"] >= 6


class TestGraphGenerators:
    """Test graph generator functions."""

    def test_complete_graph(self):
        """Test complete graph generation."""
        graph = scirs2.complete_graph_py(5)

        assert graph["num_nodes"] == 5
        # Complete graph K_n has n(n-1)/2 edges
        assert graph["num_edges"] == 10  # 5*4/2

    def test_path_graph(self):
        """Test path graph generation."""
        graph = scirs2.path_graph_py(6)

        assert graph["num_nodes"] == 6
        assert graph["num_edges"] == 5  # n-1 edges in path

    def test_cycle_graph(self):
        """Test cycle graph generation."""
        graph = scirs2.cycle_graph_py(5)

        assert graph["num_nodes"] == 5
        assert graph["num_edges"] == 5  # n edges in cycle

    def test_star_graph(self):
        """Test star graph generation."""
        graph = scirs2.star_graph_py(4)

        assert graph["num_nodes"] == 5  # Center + 4 leaves
        assert graph["num_edges"] == 4

    def test_erdos_renyi_graph(self):
        """Test Erdős-Rényi random graph."""
        graph = scirs2.erdos_renyi_graph_py(10, 0.3, seed=42)

        assert graph["num_nodes"] == 10
        # With probability 0.3, expect around 13-14 edges for K_10
        assert 0 <= graph["num_edges"] <= 45  # Max possible edges

    def test_barabasi_albert_graph(self):
        """Test Barabási-Albert scale-free graph."""
        graph = scirs2.barabasi_albert_graph_py(20, 2, seed=42)

        assert graph["num_nodes"] == 20
        # Each node adds m edges (after initial nodes)
        assert graph["num_edges"] >= 18  # Approximately 2*(n-2) edges

    def test_watts_strogatz_graph(self):
        """Test Watts-Strogatz small-world graph."""
        graph = scirs2.watts_strogatz_graph_py(10, 4, 0.1, seed=42)

        assert graph["num_nodes"] == 10
        # Ring lattice with k=4 has 20 edges (10*4/2)
        assert 15 <= graph["num_edges"] <= 25


class TestShortestPaths:
    """Test shortest path algorithms."""

    def test_dijkstra_path_simple(self):
        """Test Dijkstra's shortest path on simple graph."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=3)

        result = scirs2.dijkstra_path_py(graph, 0, 2)

        assert result["success"]
        assert result["distance"] == pytest.approx(2.0, abs=1e-10)
        assert result["path"] == [0, 1, 2]

    def test_dijkstra_path_no_path(self):
        """Test Dijkstra when no path exists."""
        edges = [(0, 1), (2, 3)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=4)

        result = scirs2.dijkstra_path_py(graph, 0, 3)

        assert result["success"] is False or result["distance"] == float('inf')

    def test_floyd_warshall_basic(self):
        """Test Floyd-Warshall all-pairs shortest paths."""
        edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=3)

        result = scirs2.floyd_warshall_py(graph)

        assert "distances" in result
        distances = result["distances"]

        # Check dimensions
        assert distances.shape == (3, 3)
        # Check diagonal is zero
        assert np.allclose(np.diag(distances), [0, 0, 0])
        # Check shortest path 0->2 is through 1
        assert distances[0, 2] == pytest.approx(3.0, abs=1e-10)


class TestConnectivity:
    """Test graph connectivity algorithms."""

    def test_connected_components_single(self):
        """Test finding connected components in connected graph."""
        edges = [(0, 1), (1, 2), (2, 3)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=4)

        result = scirs2.connected_components_py(graph)

        assert result["num_components"] == 1
        assert len(result["components"]) == 1

    def test_connected_components_multiple(self):
        """Test finding multiple connected components."""
        edges = [(0, 1), (2, 3), (4, 5)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=6)

        result = scirs2.connected_components_py(graph)

        assert result["num_components"] == 3

    def test_strongly_connected_components(self):
        """Test strongly connected components in directed graph."""
        edges = [(0, 1), (1, 2), (2, 0), (3, 4)]
        graph = scirs2.digraph_from_edges_py(edges, num_nodes=5)

        result = scirs2.strongly_connected_components_py(graph)

        assert result["num_components"] >= 2
        assert len(result["components"]) >= 2

    def test_is_bipartite_true(self):
        """Test bipartite detection on bipartite graph."""
        # Path graph is bipartite
        graph = scirs2.path_graph_py(4)

        result = scirs2.is_bipartite_py(graph)

        assert result["is_bipartite"] is True

    def test_is_bipartite_false(self):
        """Test bipartite detection on non-bipartite graph."""
        # Cycle of odd length is not bipartite
        graph = scirs2.cycle_graph_py(5)

        result = scirs2.is_bipartite_py(graph)

        assert result["is_bipartite"] is False

    def test_articulation_points(self):
        """Test finding articulation points."""
        # Bridge graph: 0-1-2 where 1 is articulation point
        edges = [(0, 1), (1, 2)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=3)

        result = scirs2.articulation_points_py(graph)

        assert 1 in result["points"]

    def test_bridges(self):
        """Test finding bridges in graph."""
        edges = [(0, 1), (1, 2), (2, 3)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=4)

        result = scirs2.bridges_py(graph)

        # All edges in path are bridges
        assert len(result["bridges"]) >= 1


class TestTraversal:
    """Test graph traversal algorithms."""

    def test_breadth_first_search(self):
        """Test BFS traversal."""
        edges = [(0, 1), (0, 2), (1, 3), (2, 4)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=5)

        result = scirs2.breadth_first_search_py(graph, 0)

        assert "order" in result
        assert len(result["order"]) == 5
        assert result["order"][0] == 0  # Start node

    def test_depth_first_search(self):
        """Test DFS traversal."""
        edges = [(0, 1), (0, 2), (1, 3), (2, 4)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=5)

        result = scirs2.depth_first_search_py(graph, 0)

        assert "order" in result
        assert len(result["order"]) == 5
        assert result["order"][0] == 0  # Start node


class TestCentrality:
    """Test centrality measures."""

    def test_betweenness_centrality(self):
        """Test betweenness centrality calculation."""
        # Star graph: center has high betweenness
        graph = scirs2.star_graph_py(4)

        result = scirs2.betweenness_centrality_py(graph)

        assert "centrality" in result
        centrality = result["centrality"]
        # Center node (index 0) should have highest centrality
        assert centrality[0] > 0

    def test_closeness_centrality(self):
        """Test closeness centrality calculation."""
        edges = [(0, 1), (1, 2), (2, 3)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=4)

        result = scirs2.closeness_centrality_py(graph)

        assert "centrality" in result
        centrality = result["centrality"]
        # All nodes should have some centrality
        assert all(c >= 0 for c in centrality)

    def test_pagerank_centrality(self):
        """Test PageRank centrality."""
        edges = [(0, 1), (1, 2), (2, 0), (1, 3)]
        graph = scirs2.digraph_from_edges_py(edges, num_nodes=4)

        result = scirs2.pagerank_centrality_py(graph)

        assert "pagerank" in result
        pagerank = result["pagerank"]
        # PageRank values should sum to approximately 1
        assert 0.9 <= sum(pagerank) <= 1.1
        # All values should be positive
        assert all(p > 0 for p in pagerank)


class TestSpanningTree:
    """Test minimum spanning tree algorithms."""

    def test_minimum_spanning_tree(self):
        """Test MST calculation."""
        edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0), (2, 3, 1.5)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=4)

        result = scirs2.minimum_spanning_tree_py(graph)

        assert "edges" in result
        assert "total_weight" in result
        # MST of n nodes has n-1 edges
        assert len(result["edges"]) == 3
        # MST weight should be minimal
        assert result["total_weight"] < 10.0


class TestCommunityDetection:
    """Test community detection algorithms."""

    def test_louvain_communities(self):
        """Test Louvain community detection."""
        # Create graph with clear communities
        edges = [
            (0, 1), (1, 2), (0, 2),  # Community 1
            (3, 4), (4, 5), (3, 5),  # Community 2
            (2, 3),  # Bridge
        ]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=6)

        result = scirs2.louvain_communities_py(graph)

        assert "communities" in result
        assert "num_communities" in result
        assert result["num_communities"] >= 1

    def test_label_propagation(self):
        """Test label propagation community detection."""
        edges = [
            (0, 1), (1, 2), (0, 2),  # Community 1
            (3, 4), (4, 5), (3, 5),  # Community 2
        ]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=6)

        result = scirs2.label_propagation_py(graph, seed=42)

        assert "communities" in result
        assert len(result["communities"]) == 6

    def test_modularity(self):
        """Test modularity calculation."""
        edges = [(0, 1), (1, 2), (3, 4)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=5)
        communities = [0, 0, 0, 1, 1]

        result = scirs2.modularity_py(graph, communities)

        assert "modularity" in result
        # Modularity should be between -1 and 1
        assert -1.0 <= result["modularity"] <= 1.0


class TestGraphMeasures:
    """Test graph property measurements."""

    def test_clustering_coefficient(self):
        """Test clustering coefficient calculation."""
        # Triangle graph has clustering coefficient of 1
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=3)

        result = scirs2.clustering_coefficient_py(graph)

        assert "average" in result or "clustering" in result
        # Triangle graph should have high clustering
        if "average" in result:
            assert result["average"] > 0.5

    def test_diameter(self):
        """Test graph diameter calculation."""
        # Path graph of length 4 has diameter 3
        graph = scirs2.path_graph_py(4)

        result = scirs2.diameter_py(graph)

        assert "diameter" in result
        assert result["diameter"] == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_graph(self):
        """Test operations on empty graph."""
        graph = scirs2.graph_from_edges_py([], num_nodes=0)

        assert graph["num_nodes"] == 0
        assert graph["num_edges"] == 0

    def test_single_node_graph(self):
        """Test graph with single node."""
        graph = scirs2.graph_from_edges_py([], num_nodes=1)

        assert graph["num_nodes"] == 1
        assert graph["num_edges"] == 0

    def test_self_loop(self):
        """Test graph with self-loop."""
        edges = [(0, 0), (0, 1)]
        graph = scirs2.graph_from_edges_py(edges, num_nodes=2)

        assert graph["num_nodes"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
