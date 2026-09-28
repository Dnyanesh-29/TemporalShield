import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import StatCard from '../components/shared/StatCard';
import AlertCard from '../components/shared/AlertCard';
import { Network, PlaySquare } from 'lucide-react';
import api from '../api/client';

export default function CommandCenterPage() {
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    total_alerts: 0,
    critical_alerts: 0,
    false_positive_rate: '0%',
    patterns_detected: 0,
    stream_status: 'PAUSED'
  });
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsData, alertsData] = await Promise.all([
          api.getStats(),
          api.getAlerts()
        ]);
        setStats(statsData);
        setAlerts(alertsData);
      } catch (err) {
        console.error("Failed to fetch command center data", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6 h-full flex flex-col gap-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Command Center</h1>
          <p className="text-gray-400 text-sm mt-1">Live overview of system detections and active alerts</p>
        </div>
      </div>

      {/* STAT ROW */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4 shrink-0">
        <StatCard label="Total Alerts" value={stats.total_alerts} trend="up" />
        <StatCard label="Critical" value={stats.critical_alerts} colorClass="text-brand-accent" trend="up" />
        <StatCard label="False Positive Rate" value={stats.false_positive_rate} colorClass="text-green-500" trend="down" />
        <StatCard label="Patterns Detected" value={stats.patterns_detected} colorClass="text-blue-400" />
        <StatCard 
          label="Stream Status" 
          value={stats.stream_status} 
          colorClass={stats.stream_status === 'LIVE' ? 'text-green-500' : 'text-gray-400'} 
        />
      </div>

      <div className="flex flex-col lg:flex-row gap-6 flex-1 min-h-0">
        
        {/* ALERT QUEUE */}
        <div className="flex-1 flex flex-col min-h-0 bg-brand-surface border border-brand-border rounded-xl">
          <div className="px-5 py-4 border-b border-brand-border">
            <h2 className="text-lg font-semibold text-white">Active Alert Queue</h2>
          </div>
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
            {alerts.map((alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}
            {alerts.length === 0 && (
              <div className="text-center text-gray-500 py-10">
                No active alerts at this time.
              </div>
            )}
          </div>
        </div>

        {/* QUICK LINKS */}
        <div className="w-full lg:w-72 flex flex-col gap-4 shrink-0">
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider px-1">Quick Links</h2>
          
          <button 
            onClick={() => navigate('/graph')}
            className="flex items-center gap-4 p-5 bg-brand-surface border border-brand-border rounded-xl hover:border-blue-500/50 hover:bg-blue-500/10 transition-colors group text-left"
          >
            <div className="bg-blue-500/20 p-3 rounded-lg">
              <Network className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <div className="font-semibold text-gray-200 group-hover:text-white">Open Graph</div>
              <div className="text-xs text-gray-500 mt-1">View full live temporal graph</div>
            </div>
          </button>

          <button 
            onClick={() => navigate('/scenarios')}
            className="flex items-center gap-4 p-5 bg-brand-surface border border-brand-border rounded-xl hover:border-brand-accent/50 hover:bg-brand-accent/10 transition-colors group text-left"
          >
            <div className="bg-brand-accent/20 p-3 rounded-lg">
              <PlaySquare className="w-6 h-6 text-brand-accent" />
            </div>
            <div>
              <div className="font-semibold text-gray-200 group-hover:text-white">Control Scenarios</div>
              <div className="text-xs text-gray-500 mt-1">Trigger demo presentations</div>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
