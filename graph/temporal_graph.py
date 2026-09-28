"""
TemporalShield / NexusTrace — Incremental Temporal Graph Engine
Maintains an in-memory live NetworkX directed multigraph updating incrementally with every event.

Node Types:
- 'employee': Bank staff (tellers, loan officers, branch managers, it admins, compliance)
- 'account': Internal bank accounts (ACC_XXXXX)
- 'counterparty': External or counterparty accounts (EXT_XXXXX)
- 'customer': Account owners

Edge Types:
- 'ACCESS': Directed edge from employee -> account with action_type, timestamp, records_accessed
- 'TRANSFER': Directed edge from account -> counterparty/account with amount, transaction_id, timestamp
- 'OWNERSHIP': Customer -> account mapping

Key Behaviors:
- Incremental updates (never rebuilt from scratch).
- Exponential time-decay weights: w = exp(-lambda * delta_t).
- Bi-directional 4-minute temporal cross-check between access events and transaction outflows.
- JSON serializable snapshot export for frontend network visualization (Vis.js / D3 / Cytoscape).
"""

import time
import math
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
import networkx as nx
import pandas as pd

logger = logging.getLogger("TemporalShield.TemporalGraph")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 4 minutes in seconds
DETECTION_WINDOW_SECONDS = 240.0
# Half-life of 2 hours for edge decay
DECAY_LAMBDA = 0.0001


class TemporalGraph:
    """
    Live incremental temporal graph for banking interactions and insider collusion detection.
    """

    def __init__(self, decay_lambda: float = DECAY_LAMBDA, window_seconds: float = DETECTION_WINDOW_SECONDS):
        self.decay_lambda = decay_lambda
        self.window_seconds = window_seconds
        self.graph = nx.MultiDiGraph()

        # In-memory fast indexes for sliding 4-minute temporal cross-checks
        # account_id -> list of recent access events: [(ts_datetime, event_dict)]
        self._recent_access: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
        # account_id -> list of recent transactions: [(ts_datetime, event_dict)]
        self._recent_transactions: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}

        # Retain last seen timestamps for time-decay reference
        self.latest_timestamp: Optional[datetime] = None

    def reset(self):
        """Resets the graph and temporal indexes."""
        self.graph.clear()
        self._recent_access.clear()
        self._recent_transactions.clear()
        self.latest_timestamp = None
        logger.info("Temporal graph and sliding buffers reset to initial state.")

    def _parse_timestamp(self, ts_raw: Any) -> datetime:
        """Parses string or datetime into a standard datetime object."""
        if isinstance(ts_raw, datetime):
            return ts_raw
        if isinstance(ts_raw, pd.Timestamp):
            return ts_raw.to_pydatetime()
        try:
            return pd.to_datetime(ts_raw).to_pydatetime()
        except Exception:
            return datetime.utcnow()

    def _update_latest_timestamp(self, ts: datetime):
        if self.latest_timestamp is None or ts > self.latest_timestamp:
            self.latest_timestamp = ts

    def compute_edge_weight(self, edge_timestamp: datetime, current_timestamp: Optional[datetime] = None) -> float:
        """
        Computes time-decay weight: w = exp(-lambda * delta_t_seconds).
        Recent interactions have weights near 1.0; older interactions decay towards 0.0.
        """
        ref_time = current_timestamp or self.latest_timestamp or datetime.utcnow()
        delta_sec = max(0.0, (ref_time - edge_timestamp).total_seconds())
        return float(math.exp(-self.decay_lambda * delta_sec))

    def add_access_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Incrementally adds an employee access event to the graph.
        
        Node updates:
        - Employee node (role, branch, last_seen)
        - Account node (account_type, last_seen)
        - Directed edge: Employee -> Account (type='ACCESS', timestamp, action_type, records)
        
        Cross-check:
        Immediately checks for recent transactions on this account within the 4-minute window.
        Returns list of any temporal correlations detected.
        """
        emp_id = str(event.get("employee_id", "EMP_UNKNOWN"))
        acct_id = str(event.get("account_id", "ACC_UNKNOWN"))
        ts = self._parse_timestamp(event.get("timestamp"))
        self._update_latest_timestamp(ts)

        # 1. Update/Add Employee Node
        if not self.graph.has_node(emp_id):
            self.graph.add_node(
                emp_id,
                id=emp_id,
                node_type="employee",
                role=event.get("role", "staff"),
                branch=event.get("branch_code", "BR_001"),
                access_count=1,
                last_seen=ts.isoformat()
            )
        else:
            self.graph.nodes[emp_id]["access_count"] = self.graph.nodes[emp_id].get("access_count", 0) + 1
            self.graph.nodes[emp_id]["last_seen"] = ts.isoformat()

        # 2. Update/Add Account Node
        if not self.graph.has_node(acct_id):
            self.graph.add_node(
                acct_id,
                id=acct_id,
                node_type="account",
                account_type=event.get("account_type", "savings"),
                branch=event.get("branch_code", "BR_001"),
                access_count=1,
                last_seen=ts.isoformat()
            )
        else:
            self.graph.nodes[acct_id]["access_count"] = self.graph.nodes[acct_id].get("access_count", 0) + 1
            self.graph.nodes[acct_id]["last_seen"] = ts.isoformat()

        # 3. Add Directed Edge
        decay_weight = self.compute_edge_weight(ts)
        self.graph.add_edge(
            emp_id,
            acct_id,
            key=f"acc_{ts.timestamp()}_{len(self.graph.get_edge_data(emp_id, acct_id) or {})}",
            edge_type="ACCESS",
            timestamp=ts.isoformat(),
            action_type=event.get("action_type", "read"),
            records_accessed=int(event.get("records_accessed", 1)),
            is_suspicious=int(event.get("is_suspicious", 0)),
            weight=decay_weight
        )

        # 4. Record in Sliding Buffer
        if acct_id not in self._recent_access:
            self._recent_access[acct_id] = []
        self._recent_access[acct_id].append((ts, event))
        # Keep only events within last 24 hours in fast index
        self._cleanup_recent_index(self._recent_access[acct_id], ts, max_age_hours=24)

        # 5. Check 4-minute Detection Window against recent transactions on this account
        correlated_events = []
        recent_txns = self._recent_transactions.get(acct_id, [])
        for txn_ts, txn_event in recent_txns:
            gap_seconds = abs((txn_ts - ts).total_seconds())
            if gap_seconds <= self.window_seconds:
                correlated_events.append({
                    "type": "TEMPORAL_LINK_CORRELATION",
                    "account_id": acct_id,
                    "employee_id": emp_id,
                    "access_event": event,
                    "transaction_event": txn_event,
                    "gap_seconds": gap_seconds,
                    "gap_minutes": round(gap_seconds / 60.0, 2),
                    "access_before_txn": ts <= txn_ts
                })

        return correlated_events

    def add_transaction_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Incrementally adds a financial transaction event to the graph.
        
        Node updates:
        - Origin Account node
        - Counterparty / Destination node
        - Directed edge: Account -> Counterparty (type='TRANSFER', amount, timestamp, etc.)
        
        Cross-check:
        Immediately checks for recent employee access on this account within the 4-minute window.
        Returns list of any temporal correlations detected.
        """
        acct_id = str(event.get("account_id", "ACC_UNKNOWN"))
        cp_id = str(event.get("counterparty_id", f"CP_{int(time.time()*1000)}"))
        ts = self._parse_timestamp(event.get("timestamp"))
        amount = float(event.get("amount", 0.0))
        txn_id = str(event.get("transaction_id", f"TXN_{int(time.time()*1000)}"))
        self._update_latest_timestamp(ts)

        # 1. Update/Add Origin Account Node
        if not self.graph.has_node(acct_id):
            self.graph.add_node(
                acct_id,
                id=acct_id,
                node_type="account",
                branch=event.get("branch_code", "BR_001"),
                total_outflow=amount,
                txn_count=1,
                last_seen=ts.isoformat()
            )
        else:
            self.graph.nodes[acct_id]["total_outflow"] = (
                self.graph.nodes[acct_id].get("total_outflow", 0.0) + amount
            )
            self.graph.nodes[acct_id]["txn_count"] = self.graph.nodes[acct_id].get("txn_count", 0) + 1
            self.graph.nodes[acct_id]["last_seen"] = ts.isoformat()

        # 2. Update/Add Counterparty Node
        if not self.graph.has_node(cp_id):
            node_type = "account" if cp_id.startswith("ACC_") else "counterparty"
            self.graph.add_node(
                cp_id,
                id=cp_id,
                node_type=node_type,
                total_inflow=amount,
                last_seen=ts.isoformat()
            )
        else:
            self.graph.nodes[cp_id]["total_inflow"] = (
                self.graph.nodes[cp_id].get("total_inflow", 0.0) + amount
            )
            self.graph.nodes[cp_id]["last_seen"] = ts.isoformat()

        # 3. Add Directed Edge
        decay_weight = self.compute_edge_weight(ts)
        self.graph.add_edge(
            acct_id,
            cp_id,
            key=txn_id,
            edge_type="TRANSFER",
            transaction_id=txn_id,
            amount=amount,
            timestamp=ts.isoformat(),
            transaction_type=event.get("transaction_type", "TRANSFER"),
            scenario=event.get("scenario", "normal"),
            is_fraud=int(event.get("is_fraud", 0)),
            weight=decay_weight
        )

        # 4. Record in Sliding Buffer
        if acct_id not in self._recent_transactions:
            self._recent_transactions[acct_id] = []
        self._recent_transactions[acct_id].append((ts, event))
        self._cleanup_recent_index(self._recent_transactions[acct_id], ts, max_age_hours=24)

        # 5. Check 4-minute Detection Window against recent employee access on this account
        correlated_events = []
        recent_acc = self._recent_access.get(acct_id, [])
        for acc_ts, acc_event in recent_acc:
            gap_seconds = abs((ts - acc_ts).total_seconds())
            if gap_seconds <= self.window_seconds:
                correlated_events.append({
                    "type": "TEMPORAL_LINK_CORRELATION",
                    "account_id": acct_id,
                    "employee_id": acc_event.get("employee_id"),
                    "access_event": acc_event,
                    "transaction_event": event,
                    "gap_seconds": gap_seconds,
                    "gap_minutes": round(gap_seconds / 60.0, 2),
                    "access_before_txn": acc_ts <= ts
                })

        return correlated_events

    def _cleanup_recent_index(self, event_list: List[Tuple[datetime, Dict[str, Any]]], current_time: datetime, max_age_hours: int = 24):
        """Prunes buffer entries older than max_age_hours to prevent memory growth."""
        cutoff = current_time - pd.Timedelta(hours=max_age_hours)
        while event_list and event_list[0][0] < cutoff:
            event_list.pop(0)

    def get_recent_transactions_for_account(self, account_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent transactions for sequence / LSTM evaluation."""
        txns = self._recent_transactions.get(account_id, [])
        return [item[1] for item in txns[-limit:]]

    def get_recent_access_for_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Returns the most recent employee access event on the account, if any."""
        acc_list = self._recent_access.get(account_id, [])
        return acc_list[-1][1] if acc_list else None

    def get_snapshot(self, max_nodes: int = 150) -> Dict[str, Any]:
        """
        Exports a graph snapshot formatted for frontend rendering (JSON compatible).
        Applies time decay to edge weights based on latest timestamp.
        """
        ref_ts = self.latest_timestamp or datetime.utcnow()

        # Rank nodes by degree to pick the most informative active subnetwork if graph is large
        if self.graph.number_of_nodes() > max_nodes:
            degrees = dict(self.graph.degree())
            top_nodes = set(sorted(degrees, key=degrees.get, reverse=True)[:max_nodes])
            sub = self.graph.subgraph(top_nodes)
        else:
            sub = self.graph

        nodes = []
        for n, data in sub.nodes(data=True):
            node_dict = dict(data)
            node_dict["id"] = n
            node_dict["label"] = n
            # Assign color / badge cues for frontend
            ntype = node_dict.get("node_type", "account")
            node_dict["group"] = ntype
            if ntype == "employee":
                node_dict["color"] = "#ff4d4f"  # Red / alert cue
                node_dict["size"] = 18
            elif ntype == "account":
                node_dict["color"] = "#1890ff"  # Blue
                node_dict["size"] = 14
            else:
                node_dict["color"] = "#faad14"  # Amber
                node_dict["size"] = 10
            nodes.append(node_dict)

        edges = []
        for u, v, k, data in sub.edges(keys=True, data=True):
            edge_dict = dict(data)
            edge_dict["from"] = u
            edge_dict["to"] = v
            edge_dict["id"] = f"{u}->{v}:{k}"
            # Recalculate dynamic decay weight
            ts_str = edge_dict.get("timestamp")
            if ts_str:
                edge_ts = self._parse_timestamp(ts_str)
                edge_dict["weight"] = self.compute_edge_weight(edge_ts, ref_ts)
            edges.append(edge_dict)

        return {
            "timestamp": ref_ts.isoformat(),
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "displayed_node_count": len(nodes),
            "displayed_edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
            "temporal_window_minutes": self.window_seconds / 60.0
        }

    def get_subgraph_around_account(self, account_id: str, radius: int = 2) -> Dict[str, Any]:
        """Returns ego-graph around an account for targeted case dossier visualization."""
        if not self.graph.has_node(account_id):
            return {"nodes": [], "edges": [], "target": account_id}

        sub_nodes = nx.single_source_shortest_path_length(self.graph.to_undirected(), account_id, cutoff=radius).keys()
        sub = self.graph.subgraph(sub_nodes)

        nodes = [dict(sub.nodes[n], id=n, label=n) for n in sub.nodes()]
        edges = []
        for u, v, k, data in sub.edges(keys=True, data=True):
            e = dict(data)
            e["from"] = u
            e["to"] = v
            e["id"] = f"{u}->{v}:{k}"
            edges.append(e)

        return {
            "target": account_id,
            "nodes": nodes,
            "edges": edges,
            "radius": radius
        }


# Quick test
if __name__ == "__main__":
    tg = TemporalGraph()
    acc_event = {
        "employee_id": "EMP_0019",
        "role": "loan_officer",
        "account_id": "ACC_10017",
        "account_type": "savings",
        "branch_code": "BR_006",
        "action_type": "read",
        "timestamp": "2026-09-01 10:00:00",
        "records_accessed": 1,
        "is_suspicious": 1
    }
    txn_event = {
        "transaction_id": "TXN_99182",
        "account_id": "ACC_10017",
        "counterparty_id": "EXT_9999",
        "amount": 48500.0,
        "transaction_type": "TRANSFER",
        "timestamp": "2026-09-01 10:02:30",
        "scenario": "temporal_link"
    }
    corrs1 = tg.add_access_event(acc_event)
    corrs2 = tg.add_transaction_event(txn_event)
    print("Correlations on txn:", corrs2)
    snapshot = tg.get_snapshot()
    print(f"Snapshot nodes: {len(snapshot['nodes'])}, edges: {len(snapshot['edges'])}")
