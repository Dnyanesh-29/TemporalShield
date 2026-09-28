import { useNavigate } from 'react-router-dom';
import SeverityBadge from './SeverityBadge';
import { Clock } from 'lucide-react';

export default function AlertCard({ alert }) {
  const navigate = useNavigate();

  // alert expects: { id, severity, pattern_name, entities: ['Emp A', 'Acc B'], time_elapsed }

  return (
    <div 
      onClick={() => navigate(`/alerts/${alert.id}`)}
      className="bg-brand-surface border border-brand-border rounded-lg p-4 cursor-pointer hover:border-gray-500 hover:bg-brand-border/30 transition-colors flex flex-col gap-3"
    >
      <div className="flex justify-between items-start">
        <SeverityBadge severity={alert.severity} />
        <div className="flex items-center text-xs text-gray-500 gap-1">
          <Clock className="w-3 h-3" />
          {alert.time_elapsed || 'Just now'}
        </div>
      </div>
      
      <div>
        <h4 className="font-semibold text-gray-100 text-sm mb-1">{alert.pattern_name}</h4>
        <div className="flex flex-wrap gap-2">
          {alert.entities?.map((entity, i) => (
            <span key={i} className="px-2 py-1 bg-gray-800 text-gray-300 rounded text-xs border border-gray-700">
              {entity}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
