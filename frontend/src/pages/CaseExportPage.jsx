import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Filter, Download, ExternalLink, Calendar } from 'lucide-react';
import api from '../api/client';
import SeverityBadge from '../components/shared/SeverityBadge';

export default function CaseExportPage() {
  const [cases, setCases] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const navigate = useNavigate();

  useEffect(() => {
    // For demo purposes, we reuse getAlerts to populate the case history
    const loadCases = async () => {
      try {
        const data = await api.getAlerts();
        // Duplicate some to make the table look full
        const extended = [
          ...data,
          { id: 'ALT-1039', severity: 'MEDIUM', pattern_name: 'Role Mismatch', entities: ['Emp: E-12', 'Acc: A-55'], time_elapsed: '2 hrs ago' },
          { id: 'ALT-1038', severity: 'LOW', pattern_name: 'Off-Hours Access', entities: ['Emp: E-99'], time_elapsed: '1 day ago' },
          { id: 'ALT-1037', severity: 'CRITICAL', pattern_name: 'Circular Transfer', entities: ['Acc: A-1', 'Acc: A-2', 'Acc: A-3'], time_elapsed: '2 days ago' },
        ];
        setCases(extended);
      } catch (err) {
        console.error("Failed to load cases", err);
      }
    };
    loadCases();
  }, []);

  const handleExportJSON = (caseObj) => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(caseObj, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `case_export_${caseObj.id}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const handleExportAll = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(filteredCases, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `bulk_export.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const filteredCases = cases.filter(c => {
    const matchesSearch = c.id.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          c.pattern_name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSeverity = severityFilter === 'ALL' || c.severity === severityFilter;
    return matchesSearch && matchesSeverity;
  });

  return (
    <div className="p-6 h-full flex flex-col gap-6 overflow-hidden">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Case Archive & Export</h1>
        <p className="text-gray-400 text-sm mt-1">Review investigated alerts and generate regulatory submission packages</p>
      </div>

      {/* FILTER ROW */}
      <div className="bg-brand-surface border border-brand-border rounded-xl p-4 flex flex-col md:flex-row gap-4 justify-between items-center shrink-0">
        <div className="flex gap-4 w-full md:w-auto">
          {/* Search */}
          <div className="relative flex-1 md:w-64">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input 
              type="text" 
              placeholder="Search ID or Pattern..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-brand-bg border border-brand-border text-sm text-gray-200 rounded-lg pl-9 pr-4 py-2 focus:outline-none focus:border-brand-accent transition-colors"
            />
          </div>

          {/* Severity Dropdown */}
          <div className="relative">
            <Filter className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <select 
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="appearance-none bg-brand-bg border border-brand-border text-sm text-gray-200 rounded-lg pl-9 pr-8 py-2 focus:outline-none focus:border-brand-accent transition-colors cursor-pointer"
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MEDIUM">Medium</option>
              <option value="LOW">Low</option>
            </select>
          </div>
        </div>

        <button 
          onClick={handleExportAll}
          className="flex items-center gap-2 bg-gray-200 hover:bg-white text-black px-4 py-2 rounded-lg font-semibold text-sm transition-colors w-full md:w-auto justify-center"
        >
          <Download className="w-4 h-4" /> Export Filtered Cases
        </button>
      </div>

      {/* CASES TABLE */}
      <div className="flex-1 bg-brand-surface border border-brand-border rounded-xl overflow-hidden flex flex-col">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-gray-300">
            <thead className="text-xs text-gray-500 uppercase bg-brand-bg border-b border-brand-border">
              <tr>
                <th className="px-6 py-4 font-semibold">Alert ID</th>
                <th className="px-6 py-4 font-semibold">Pattern</th>
                <th className="px-6 py-4 font-semibold">Severity</th>
                <th className="px-6 py-4 font-semibold">Involved Entities</th>
                <th className="px-6 py-4 font-semibold">Timestamp</th>
                <th className="px-6 py-4 font-semibold text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredCases.map((caseObj) => (
                <tr key={caseObj.id} className="border-b border-brand-border hover:bg-brand-border/30 transition-colors">
                  <td className="px-6 py-4 font-mono text-gray-100">{caseObj.id}</td>
                  <td className="px-6 py-4 font-medium">{caseObj.pattern_name}</td>
                  <td className="px-6 py-4">
                    <SeverityBadge severity={caseObj.severity} />
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex flex-wrap gap-1">
                      {caseObj.entities.map((ent, i) => (
                         <span key={i} className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded border border-gray-700">
                           {ent}
                         </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-gray-500 whitespace-nowrap">
                    <div className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" /> {caseObj.time_elapsed}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button 
                        onClick={() => navigate(`/alerts/${caseObj.id}`)}
                        className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors tooltip"
                        title="View Details"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </button>
                      <button 
                        onClick={() => handleExportJSON(caseObj)}
                        className="p-2 text-blue-400 hover:text-blue-300 hover:bg-blue-900/30 rounded transition-colors"
                        title="Export JSON"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {filteredCases.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              No cases match your filters.
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
