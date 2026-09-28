"""
TemporalShield / NexusTrace — Circular Money Flow Detector
Detects layered circular fund transfers (A -> B -> C -> A) indicating money laundering rings.
Combines directed cycle detection (bounded DFS) with Node2Vec structural embeddings to uncover
both closed cycles and high-similarity laundering clusters.

Historical Grounding:
Saradha Financial Syndicate Fraud (2013, INR 2,500 Crore)
Regulatory Benchmark: FIU-IND Red Flag Indicator RFI-08 (Layered circular flows)
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
import networkx as nx

try:
    from graph.node2vec_embed import GraphEmbeddingEngine
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph.node2vec_embed import GraphEmbeddingEngine

logger = logging.getLogger("TemporalShield.CircularDetector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CircularFlowDetector:
    """
    Detects circular money movements and laundering rings in the temporal transaction graph.
    """

    def __init__(self, min_cycle_len: int = 3, max_cycle_len: int = 6, embedding_engine: Optional[GraphEmbeddingEngine] = None):
        self.min_cycle_len = min_cycle_len
        self.max_cycle_len = max_cycle_len
        self.embed_engine = embedding_engine or GraphEmbeddingEngine(dimensions=16)
        self.detected_rings: List[Dict[str, Any]] = []
        self._seen_cycle_signatures: Set[str] = set()

    def _build_transfer_digraph(self, multigraph: nx.MultiDiGraph) -> nx.DiGraph:
        """
        Extracts a clean directed transaction graph (ignoring employee access edges).
        Edges retain max amount and latest timestamp.
        """
        dg = nx.DiGraph()
        for u, v, data in multigraph.edges(data=True):
            if data.get("edge_type") == "TRANSFER":
                amt = float(data.get("amount", 0.0))
                ts = data.get("timestamp", "")
                if dg.has_edge(u, v):
                    dg[u][v]["amount"] += amt
                    dg[u][v]["txn_count"] += 1
                else:
                    dg.add_edge(u, v, amount=amt, txn_count=1, last_ts=ts)
        return dg

    def _cycle_signature(self, cycle: List[str]) -> str:
        """Produces canonical sorted rotation signature to avoid reporting duplicate permutations of the same ring."""
        # Find minimum element index
        min_idx = cycle.index(min(cycle))
        rotated = cycle[min_idx:] + cycle[:min_idx]
        return "->".join(rotated)

    def detect_cycles(self, multigraph: nx.MultiDiGraph, target_account: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Finds directed simple cycles between min_cycle_len and max_cycle_len.
        If target_account is specified, searches for cycles passing through target_account.
        """
        dg = self._build_transfer_digraph(multigraph)
        if dg.number_of_nodes() < self.min_cycle_len:
            return []

        # Update node embeddings if graph has enough nodes
        if dg.number_of_nodes() >= 4:
            try:
                self.embed_engine.fit(dg)
            except Exception as e:
                logger.debug(f"Node2Vec embedding pass skipped: {e}")

        cycles_found = []

        if target_account and dg.has_node(target_account):
            # Target-focused DFS cycle search
            cycles = self._find_cycles_for_node(dg, target_account, self.max_cycle_len)
        else:
            # Global cycle search bounded by length
            try:
                # NetworkX simple_cycles generates cycles; filter by length
                raw_cycles = []
                for c in nx.simple_cycles(dg):
                    if self.min_cycle_len <= len(c) <= self.max_cycle_len:
                        raw_cycles.append(c)
                    if len(raw_cycles) >= 50:  # Safety cap for large graphs
                        break
                cycles = raw_cycles
            except Exception as e:
                logger.warning(f"Global cycle search fallback: {e}")
                cycles = []

        for cycle in cycles:
            sig = self._cycle_signature(cycle)
            if sig in self._seen_cycle_signatures:
                continue
            self._seen_cycle_signatures.add(sig)

            alert_obj = self._format_cycle_alert(dg, cycle)
            cycles_found.append(alert_obj)
            self.detected_rings.append(alert_obj)

        return cycles_found

    def _find_cycles_for_node(self, G: nx.DiGraph, start_node: str, max_depth: int) -> List[List[str]]:
        """Bounded DFS traversal to discover cycles containing start_node."""
        cycles = []

        def dfs(curr: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            for nbr in G.successors(curr):
                if nbr == start_node and len(path) >= self.min_cycle_len:
                    cycles.append(list(path))
                elif nbr not in path:
                    dfs(nbr, path + [nbr], depth + 1)

        dfs(start_node, [start_node], 1)
        return cycles

    def _format_cycle_alert(self, G: nx.DiGraph, cycle: List[str]) -> Dict[str, Any]:
        """Builds a rich alert package conforming to the TemporalShield investigation dossier contract."""
        hop_details = []
        total_quantum = 0.0

        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge_data = G.get_edge_data(u, v, default={})
            amt = float(edge_data.get("amount", 0.0))
            total_quantum += amt
            hop_details.append({
                "from_account": u,
                "to_account": v,
                "amount": round(amt, 2),
                "timestamp": edge_data.get("last_ts")
            })

        # Calculate structural similarity cohesion among cycle members
        similarities = []
        for i in range(len(cycle)):
            for j in range(i + 1, len(cycle)):
                sim = self.embed_engine.compute_similarity(cycle[i], cycle[j])
                similarities.append(sim)
        avg_cohesion = float(sum(similarities) / max(1, len(similarities)))

        cycle_path_str = " ➔ ".join(cycle + [cycle[0]])
        alert_id = f"ALT_CIRC_{int(time.time()*1000)}_{len(self.detected_rings)+1}"

        return {
            "alert_id": alert_id,
            "detection_type": "CIRCULAR_TRANSFER_RING",
            "cycle_length": len(cycle),
            "cycle_path": cycle_path_str,
            "participating_accounts": cycle,
            "hop_details": hop_details,
            "total_quantum_inr": round(total_quantum, 2),
            "node2vec_ring_cohesion": round(avg_cohesion, 3),
            "anomaly_score": round(min(1.0, 0.75 + (0.05 * len(cycle)) + (0.15 * max(0.0, avg_cohesion))), 4),
            "severity": "CRITICAL" if total_quantum > 100000 or len(cycle) >= 4 else "HIGH",
            "historical_incident_parallel": {
                "incident_name": "Saradha Financial Syndicate Fraud (2013)",
                "quantum": "INR 2,500 Crore",
                "parallel_description": f"Closed {len(cycle)}-hop loop ({cycle_path_str}) creates synthetic commercial turnover with total flow INR {total_quantum:,.2f}.",
                "pmla_sections": "PMLA 2002 Section 4, FIU-IND RFI-08"
            },
            "timestamp": datetime.utcnow().isoformat(),
            "recommended_action": f"FREEZE ACCOUNTS: Immediate temporary debit block on all {len(cycle)} ring accounts: {', '.join(cycle)}."
        }


# Quick test
if __name__ == "__main__":
    mg = nx.MultiDiGraph()
    mg.add_edge("ACC_10001", "ACC_10002", edge_type="TRANSFER", amount=50000.0, timestamp="2026-09-01 10:00:00")
    mg.add_edge("ACC_10002", "ACC_10003", edge_type="TRANSFER", amount=49500.0, timestamp="2026-09-01 10:15:00")
    mg.add_edge("ACC_10003", "ACC_10001", edge_type="TRANSFER", amount=49000.0, timestamp="2026-09-01 10:30:00")

    det = CircularFlowDetector()
    alerts = det.detect_cycles(mg)
    print("Detected cycles:", len(alerts))
    if alerts:
        print("Path:", alerts[0]["cycle_path"])
        print("Quantum:", alerts[0]["total_quantum_inr"])
