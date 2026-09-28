import { useState, useEffect } from 'react';
import { Play, RotateCcw, Activity, PauseCircle } from 'lucide-react';
import api from '../api/client';
import AlertCard from '../components/shared/AlertCard';

export default function ScenarioControlPage() {
  const [scenarios, setScenarios] = useState([]);
  const [streamStatus, setStreamStatus] = useState('PAUSED');
  const [activeScenario, setActiveScenario] = useState(null);
  const [activeTimer, setActiveTimer] = useState(0);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    // Load initial data
    const loadData = async () => {
      try {
        const scens = await api.getScenarios();
        setScenarios(scens);
      } catch (err) {
        console.error("Failed to load scenarios", err);
      }
    };
    loadData();
  }, []);

  useEffect(() => {
    // Polling for stream status and live alerts feed
    const pollSystem = async () => {
      try {
        const [stats, liveAlerts] = await Promise.all([
          api.getStats(),
          api.getAlerts()
        ]);
        setStreamStatus(stats.stream_status);
        setAlerts(liveAlerts);
      } catch (err) {
        console.error("Failed to poll system status", err);
      }
    };

    pollSystem();
    const interval = setInterval(pollSystem, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Timer for active scenario
    let timerInterval;
    if (activeScenario) {
      timerInterval = setInterval(() => {
        setActiveTimer((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(timerInterval);
  }, [activeScenario]);

  const handleTrigger = async (id) => {
    try {
      await api.triggerScenario(id);
      setActiveScenario(id);
      setActiveTimer(0);
      setStreamStatus('LIVE'); // Optimistically update
    } catch (err) {
      console.error("Failed to trigger scenario", err);
    }
  };

  const handleReset = async () => {
    try {
      await api.resetGraph();
      setActiveScenario(null);
      setActiveTimer(0);
      setStreamStatus('PAUSED'); // Optimistically update
    } catch (err) {
      console.error("Failed to reset", err);
    }
  };

  // Format timer into MM:SS
  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="p-6 h-full flex flex-col gap-6 overflow-y-auto">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Scenario Control Room</h1>
        <p className="text-gray-400 text-sm mt-1">Interactive demo environment for judges to evaluate detection models</p>
      </div>

      {/* STREAM STATUS BAR */}
      <div className="bg-brand-surface border border-brand-border rounded-xl p-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Stream Status</span>
          
          {streamStatus === 'LIVE' ? (
            <div className="flex items-center gap-2 bg-green-500/10 border border-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-bold tracking-wider">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
              </span>
              LIVE
            </div>
          ) : (
            <div className="flex items-center gap-2 bg-gray-500/10 border border-gray-500/20 text-gray-400 px-3 py-1 rounded-full text-sm font-bold tracking-wider">
              <PauseCircle className="w-4 h-4" />
              PAUSED
            </div>
          )}
        </div>

        <button 
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2 bg-brand-bg border border-brand-border hover:bg-gray-800 hover:text-white text-gray-300 rounded-lg transition-colors text-sm font-semibold"
        >
          <RotateCcw className="w-4 h-4" />
          Reset Stream
        </button>
      </div>

      {/* SCENARIO CARDS (2x2 grid) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 shrink-0">
        {scenarios.map((scenario) => {
          const isActive = activeScenario === scenario.id;
          
          return (
            <div 
              key={scenario.id}
              className={`bg-brand-surface rounded-xl p-5 border transition-all flex flex-col justify-between h-48 ${
                isActive 
                  ? 'border-brand-accent shadow-[0_0_15px_rgba(192,57,43,0.3)]' 
                  : 'border-brand-border hover:border-gray-600'
              }`}
            >
              <div>
                <div className="flex justify-between items-start mb-2">
                  <h3 className={`font-bold text-lg ${scenario.type === 'clean' ? 'text-green-400' : 'text-gray-100'}`}>
                    {scenario.title}
                  </h3>
                  {isActive && (
                    <div className="text-brand-accent font-mono text-sm font-bold bg-brand-accent/10 px-2 py-1 rounded">
                      {formatTime(activeTimer)}
                    </div>
                  )}
                </div>
                <p className="text-sm text-gray-400 leading-relaxed">
                  {scenario.description}
                </p>
              </div>

              <div className="mt-4">
                <button
                  onClick={() => handleTrigger(scenario.id)}
                  disabled={isActive}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-bold transition-colors ${
                    isActive 
                      ? 'bg-brand-accent/20 text-brand-accent cursor-not-allowed border border-brand-accent/50' 
                      : 'bg-blue-600 hover:bg-blue-700 text-white border border-blue-500'
                  }`}
                >
                  <Play className="w-4 h-4 fill-current" />
                  {isActive ? 'Running...' : 'Trigger Scenario'}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* LIVE ALERT FEED */}
      <div className="flex-1 min-h-[300px] flex flex-col bg-brand-surface border border-brand-border rounded-xl mt-2">
        <div className="px-5 py-4 border-b border-brand-border flex items-center justify-between bg-brand-bg/50 rounded-t-xl">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-gray-400" />
            Live Alert Feed
          </h2>
          <span className="text-xs text-gray-500">Updates as scenario runs</span>
        </div>
        
        <div className="flex-1 overflow-y-auto p-5 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {alerts.map((alert) => (
            <AlertCard key={alert.id} alert={alert} />
          ))}
          {alerts.length === 0 && (
            <div className="col-span-full flex flex-col items-center justify-center text-gray-500 py-10 h-full">
              <Activity className="w-10 h-10 mb-3 opacity-20" />
              <p>Waiting for scenario to generate alerts...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
