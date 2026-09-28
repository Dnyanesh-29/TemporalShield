import { NavLink, useNavigate } from 'react-router-dom';
import { LayoutDashboard, Network, PlaySquare, Archive, LogOut, Bell } from 'lucide-react';
import { useState, useEffect, useContext } from 'react';
import { RoleContext } from '../App';
import api from '../api/client';

export default function Sidebar() {
  const { role, setRole } = useContext(RoleContext);
  const navigate = useNavigate();
  const [alertCount, setAlertCount] = useState(0);

  useEffect(() => {
    // Poll for total active alerts
    const fetchAlerts = async () => {
      try {
        const alerts = await api.getAlerts();
        setAlertCount(alerts.length);
      } catch (err) {
        console.error("Failed to fetch alerts", err);
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleLogout = () => {
    setRole(null);
    navigate('/');
  };

  const navItems = [
    { name: 'Command Center', path: '/dashboard', icon: LayoutDashboard, badge: alertCount },
    { name: 'Graph Explorer', path: '/graph', icon: Network },
    { name: 'Scenario Control', path: '/scenarios', icon: PlaySquare },
    { name: 'Case Export', path: '/cases', icon: Archive },
  ];

  return (
    <aside className="w-64 border-r border-brand-border bg-brand-surface flex flex-col justify-between shrink-0 h-full">
      <div className="p-4">
        
        {/* User Context */}
        <div className="mb-6 px-4 py-3 bg-brand-bg rounded-lg border border-brand-border">
          <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold mb-1">Active Role</div>
          <div className="text-sm text-gray-200 font-medium">{role || 'Not Selected'}</div>
        </div>

        {/* Navigation */}
        <nav className="flex flex-col gap-2">
          {navItems.map((item) => (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) => 
                `flex items-center justify-between px-4 py-3 rounded-lg transition-colors ${
                  isActive 
                  ? 'bg-brand-accent/20 text-brand-accent font-medium' 
                  : 'text-gray-400 hover:bg-brand-border/30 hover:text-gray-200'
                }`
              }
            >
              <div className="flex items-center gap-3">
                <item.icon className="w-5 h-5" />
                <span className="text-sm">{item.name}</span>
              </div>
              
              {/* Badge for Command Center */}
              {item.badge > 0 && (
                <div className="bg-brand-accent text-white text-[10px] font-bold px-2 py-0.5 rounded-full">
                  {item.badge}
                </div>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="p-4 border-t border-brand-border">
        <button 
          onClick={handleLogout}
          className="flex items-center gap-3 px-4 py-3 w-full text-gray-400 hover:text-white hover:bg-brand-border/30 rounded-lg transition-colors"
        >
          <LogOut className="w-5 h-5" />
          <span className="text-sm">Change Role</span>
        </button>
      </div>
    </aside>
  );
}
