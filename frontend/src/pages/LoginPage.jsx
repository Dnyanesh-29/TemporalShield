import { useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { RoleContext } from '../App';
import { Shield, Search, UserCog, Play } from 'lucide-react';

export default function LoginPage() {
  const { setRole } = useContext(RoleContext);
  const navigate = useNavigate();

  const handleRoleSelect = (selectedRole) => {
    setRole(selectedRole);
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-brand-bg p-4">
      <div className="max-w-md w-full bg-brand-surface border border-brand-border rounded-xl shadow-2xl p-8 flex flex-col items-center">
        
        {/* Logo Section */}
        <div className="flex items-center gap-3 mb-4">
          <div className="bg-brand-accent p-2 rounded-lg">
            <Shield className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Nexus<span className="text-brand-accent">Trace</span>
          </h1>
        </div>
        
        {/* Tagline */}
        <p className="text-gray-400 text-center mb-10 text-sm leading-relaxed">
          Closing the 4-minute gap between insider access and financial fraud
        </p>

        {/* Role Selection Buttons */}
        <div className="flex flex-col w-full gap-4">
          <button 
            onClick={() => handleRoleSelect('Investigator')}
            className="flex items-center gap-4 p-4 rounded-lg border border-brand-border bg-brand-bg hover:border-brand-accent hover:bg-brand-accent/10 transition-colors group cursor-pointer"
          >
            <div className="bg-brand-surface p-2 rounded-md group-hover:bg-brand-accent/20 transition-colors">
              <Search className="w-5 h-5 text-gray-300 group-hover:text-brand-accent" />
            </div>
            <div className="text-left flex-1">
              <div className="font-semibold text-gray-200 group-hover:text-white">Investigator</div>
              <div className="text-xs text-gray-500 mt-1">Analyze alerts & graph timelines</div>
            </div>
          </button>

          <button 
            onClick={() => handleRoleSelect('Admin')}
            className="flex items-center gap-4 p-4 rounded-lg border border-brand-border bg-brand-bg hover:border-brand-accent hover:bg-brand-accent/10 transition-colors group cursor-pointer"
          >
            <div className="bg-brand-surface p-2 rounded-md group-hover:bg-brand-accent/20 transition-colors">
              <UserCog className="w-5 h-5 text-gray-300 group-hover:text-brand-accent" />
            </div>
            <div className="text-left flex-1">
              <div className="font-semibold text-gray-200 group-hover:text-white">Admin</div>
              <div className="text-xs text-gray-500 mt-1">System management & users</div>
            </div>
          </button>

          <button 
            onClick={() => handleRoleSelect('Demo Mode')}
            className="flex items-center gap-4 p-4 rounded-lg border border-brand-border bg-brand-bg hover:border-brand-accent hover:bg-brand-accent/10 transition-colors group cursor-pointer"
          >
            <div className="bg-brand-surface p-2 rounded-md group-hover:bg-brand-accent/20 transition-colors">
              <Play className="w-5 h-5 text-gray-300 group-hover:text-brand-accent" />
            </div>
            <div className="text-left flex-1">
              <div className="font-semibold text-gray-200 group-hover:text-white">Demo Mode</div>
              <div className="text-xs text-gray-500 mt-1">Trigger scenarios for judges</div>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
