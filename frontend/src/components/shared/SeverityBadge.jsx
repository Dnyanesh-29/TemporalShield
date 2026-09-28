export default function SeverityBadge({ severity }) {
  const normalized = severity?.toUpperCase() || 'LOW';
  
  let bgClass = 'bg-gray-500/10 text-gray-400 border-gray-500/20';
  let pulse = false;

  switch (normalized) {
    case 'CRITICAL':
      bgClass = 'bg-brand-accent/20 text-red-500 border-brand-accent/50';
      pulse = true;
      break;
    case 'HIGH':
      bgClass = 'bg-orange-500/20 text-orange-400 border-orange-500/50';
      break;
    case 'MEDIUM':
      bgClass = 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      break;
    case 'LOW':
      bgClass = 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      break;
  }

  return (
    <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${bgClass}`}>
      {pulse && (
        <span className="flex w-2 h-2 mr-1.5 relative">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full w-2 h-2 bg-red-500"></span>
        </span>
      )}
      {normalized}
    </div>
  );
}
