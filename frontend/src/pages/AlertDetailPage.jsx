import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Download, X, Clock, ArrowRight, ShieldAlert, BarChart3, Fingerprint, Link as LinkIcon } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import api from '../api/client';
import SeverityBadge from '../components/shared/SeverityBadge';

export default function AlertDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [evidence, setEvidence] = useState(null);

  useEffect(() => {
    const fetchEvidence = async () => {
      try {
        const data = await api.getAlertEvidence(id);
        setEvidence(data);
      } catch (err) {
        console.error("Failed to fetch evidence", err);
      }
    };
    fetchEvidence();
  }, [id]);

  if (!evidence) {
    return <div className="p-8 text-gray-400">Loading evidence package...</div>;
  }

  const handleExport = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(evidence, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `evidence_${evidence.id}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  return (
    <div className="p-6 h-full flex flex-col gap-6 overflow-y-auto">
      
      {/* HEADER SECTION */}
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-2xl font-bold text-white tracking-tight">{evidence.pattern_name}</h1>
            <SeverityBadge severity={evidence.severity} />
          </div>
          <p className="text-gray-400 text-sm max-w-2xl leading-relaxed">
            {evidence.explanation}
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm font-semibold text-gray-500 uppercase">Alert ID</div>
          <div className="text-lg font-mono text-gray-200">{evidence.id}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* LEFT COLUMN */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          
          {/* ENTITY ROW */}
          <div className="bg-brand-surface border border-brand-border rounded-xl p-5">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Fingerprint className="w-4 h-4" /> Involved Entities
            </h2>
            <div className="flex flex-wrap gap-3">
              {evidence.entities.map((entity, i) => (
                <div key={i} className="flex items-center gap-2 px-3 py-2 bg-brand-bg border border-gray-700 rounded-lg hover:border-gray-500 transition-colors cursor-pointer">
                  <span className="text-xs text-gray-500 uppercase">{entity.type}</span>
                  <span className="text-sm font-semibold text-gray-200">{entity.label}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-3 italic">* Click an entity to highlight it on the temporal graph</p>
          </div>

          {/* TIMELINE */}
          <div className="bg-brand-surface border border-brand-border rounded-xl p-5">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-6 flex items-center gap-2">
              <Clock className="w-4 h-4" /> Access → Transaction Timeline
            </h2>
            <div className="flex items-center justify-between bg-brand-bg border border-brand-border p-4 rounded-lg relative">
              <div className="absolute top-1/2 left-8 right-8 h-0.5 bg-gray-700 -translate-y-1/2 z-0"></div>
              
              <div className="relative z-10 flex flex-col items-center bg-brand-bg px-2">
                <div className="w-3 h-3 rounded-full bg-blue-500 mb-2"></div>
                <div className="text-sm font-bold text-gray-200">Access</div>
                <div className="text-xs text-gray-500">{evidence.timeline.access_time}</div>
              </div>

              <div className="relative z-10 flex flex-col items-center">
                <div className="bg-brand-accent/20 border border-brand-accent text-red-400 px-3 py-1 rounded-full text-xs font-bold mb-6">
                  {evidence.timeline.gap_minutes} Min Gap
                </div>
              </div>

              <div className="relative z-10 flex flex-col items-center bg-brand-bg px-2">
                <div className="w-3 h-3 rounded-full bg-amber-500 mb-2"></div>
                <div className="text-sm font-bold text-gray-200">Transaction</div>
                <div className="text-xs text-gray-500">{evidence.timeline.txn_time}</div>
              </div>
            </div>
          </div>

          {/* SHAP WATERFALL CHART */}
          <div className="bg-brand-surface border border-brand-border rounded-xl p-5">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" /> Feature Contributions (SHAP Values)
            </h2>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={evidence.shap_values} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                  <XAxis type="number" hide />
                  <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fill: '#9ca3af', fontSize: 12 }} width={140} />
                  <Tooltip 
                    cursor={{ fill: '#2a2a2a' }}
                    contentStyle={{ backgroundColor: '#1A1A1A', border: '1px solid #333', borderRadius: '8px' }}
                    itemStyle={{ color: '#C0392B' }}
                  />
                  <Bar dataKey="contribution" radius={[0, 4, 4, 0]} barSize={20}>
                    {evidence.shap_values.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={index === 0 ? '#C0392B' : '#4b5563'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN */}
        <div className="flex flex-col gap-6">
          
          {/* CONFIDENCE SCORE */}
          <div className="bg-brand-surface border border-brand-border rounded-xl p-5 text-center">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">ML Confidence Score</h2>
            <div className="relative inline-flex items-center justify-center">
              <svg className="w-32 h-32 transform -rotate-90">
                <circle cx="64" cy="64" r="56" className="text-gray-700 stroke-current" strokeWidth="12" fill="transparent" />
                <circle 
                  cx="64" 
                  cy="64" 
                  r="56" 
                  className="text-brand-accent stroke-current" 
                  strokeWidth="12" 
                  fill="transparent" 
                  strokeDasharray="351"
                  strokeDashoffset={351 - (351 * evidence.confidence_score) / 100}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-3xl font-bold text-white">{evidence.confidence_score}%</span>
              </div>
            </div>
          </div>

          {/* LINKED INDIAN INCIDENT */}
          <div className="bg-brand-surface border border-brand-border rounded-xl p-5">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <LinkIcon className="w-4 h-4" /> Historical Precedent
            </h2>
            <div className="bg-gray-800/50 border border-gray-700 p-4 rounded-lg">
              <div className="text-gray-300 font-medium mb-1">{evidence.linked_incident.name}</div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-500">Year: {evidence.linked_incident.year}</span>
                <span className="text-red-400 font-bold">{evidence.linked_incident.amount}</span>
              </div>
            </div>
          </div>

          {/* ACTION BUTTONS */}
          <div className="flex flex-col gap-3 mt-auto">
            <button 
              onClick={handleExport}
              className="flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-lg font-semibold transition-colors w-full"
            >
              <Download className="w-5 h-5" /> Export Evidence JSON
            </button>
            <button 
              onClick={() => navigate(-1)}
              className="flex items-center justify-center gap-2 bg-brand-surface border border-brand-border hover:bg-gray-800 text-gray-300 p-3 rounded-lg font-semibold transition-colors w-full"
            >
              <X className="w-5 h-5" /> Close Details
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}
