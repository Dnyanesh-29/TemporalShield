import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

export const api = {
  getAlerts: async () => {
    return [
      { id: 'ALT-1042', severity: 'CRITICAL', pattern_name: 'Temporal Link', entities: ['Emp: E-902', 'Acc: A-4421'], time_elapsed: '2 min ago' },
      { id: 'ALT-1041', severity: 'HIGH', pattern_name: 'Structuring', entities: ['Acc: A-1192'], time_elapsed: '14 min ago' }
    ];
  },
  getAlertDetail: async (id) => { return null; },
  getAlertEvidence: async (id) => {
    // Mock complete evidence package
    return {
      id: id,
      pattern_name: 'Temporal Link (Access -> Transfer)',
      severity: 'CRITICAL',
      explanation: 'An employee accessed an account outside normal parameters, and a high-value transfer left that same account 4 minutes later.',
      confidence_score: 94,
      entities: [
        { type: 'Employee', label: 'E-902 (Branch Manager)' },
        { type: 'Account', label: 'A-4421 (Savings)' },
        { type: 'Transaction', label: 'Tx-9 (₹2,00,000)' }
      ],
      shap_values: [
        { feature: 'Time gap (4 min)', contribution: 0.45 },
        { feature: 'Role access mismatch', contribution: 0.25 },
        { feature: 'Amount deviation', contribution: 0.15 },
        { feature: 'Off-hours access', contribution: 0.05 },
      ],
      timeline: {
        access_time: '09:03 AM',
        txn_time: '09:07 AM',
        gap_minutes: 4
      },
      linked_incident: {
        name: 'Citibank India Fraud',
        year: '2010',
        amount: '₹400 Crore'
      }
    };
  },
  getGraph: async () => { return { nodes: [], edges: [] }; },
  triggerScenario: async (scenarioName) => { return { success: true }; },
  getScenarios: async () => {
    return [
      { id: 'temporal_link', title: 'Suspicious — Temporal Link', description: 'Employee accesses account, transfer follows in 4 minutes', type: 'suspicious' },
      { id: 'circular_transfer', title: 'Suspicious — Circular Transfer', description: 'A→B→C→A money laundering ring', type: 'suspicious' },
      { id: 'structuring', title: 'Suspicious — Structuring', description: '5x sub-₹50K transfers in 2 hrs', type: 'suspicious' },
      { id: 'clean_busy', title: 'Clean — Busy Branch Day', description: 'High volume but legitimate. Must NOT trigger alerts.', type: 'clean' }
    ];
  },
  getStats: async () => {
    return { total_alerts: 124, critical_alerts: 3, false_positive_rate: '4.2%', patterns_detected: 5, stream_status: 'LIVE', scenario_name: 'None' };
  },
  resetGraph: async () => { return { success: true }; }
};

export default api;
