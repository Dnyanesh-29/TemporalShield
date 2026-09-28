import { Shield, Activity, PauseCircle } from 'lucide-react';
import { useState, useEffect } from 'react';
import api from '../api/client';

export default function Navbar() {
  const [streamStatus, setStreamStatus] = useState('LIVE');
  const [scenarioName, setScenarioName] = useState('Default Monitoring');

  useEffect(() => {
    // Poll for stats every 5 seconds as specified in the PDF for polling
    const fetchStatus = async () => {
      try {
        const stats = await api.getStats();
        if (stats.stream_status) setStreamStatus(stats.stream_status);
        if (stats.scenario_name) setScenarioName(stats.scenario_name);
      } catch (err) {
        console.error("Failed to fetch stream status", err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <nav className="h-16 border-b border-brand-border bg-brand-surface px-6 flex items-center justify-between shrink-0">
      
      {/* Logo */}
      <div className="flex items-center gap-2">
        <Shield className="w-6 h-6 text-brand-accent" />
        <span className="text-xl font-bold tracking-tight text-white">
          Nexus<span className="text-brand-accent">Trace</span>
        </span>
      </div>

      {/* Center - Scenario Name */}
      <div className="absolute left-1/2 -translate-x-1/2 hidden md:block">
        <span className="text-sm font-medium text-gray-400">
          Scenario: <span className="text-gray-200">{scenarioName}</span>
        </span>
      </div>

      {/* Right - Stream Status */}
      <div className="flex items-center gap-2">
        {streamStatus === 'LIVE' ? (
          <div className="flex items-center gap-2 bg-green-500/10 border border-green-500/20 text-green-400 px-3 py-1 rounded-full text-xs font-semibold tracking-wider">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            LIVE
          </div>
        ) : (
          <div className="flex items-center gap-2 bg-gray-500/10 border border-gray-500/20 text-gray-400 px-3 py-1 rounded-full text-xs font-semibold tracking-wider">
            <PauseCircle className="w-3 h-3" />
            PAUSED
          </div>
        )}
      </div>

    </nav>
  );
}
