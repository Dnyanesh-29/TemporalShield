import { Routes, Route, useLocation } from 'react-router-dom';
import { useState, createContext } from 'react';

// Pages
import LoginPage from './pages/LoginPage';
import CommandCenterPage from './pages/CommandCenterPage';
import GraphExplorerPage from './pages/GraphExplorerPage';
import AlertDetailPage from './pages/AlertDetailPage';
import ScenarioControlPage from './pages/ScenarioControlPage';
import CaseExportPage from './pages/CaseExportPage';

// Components
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';

// Global Role Context
export const RoleContext = createContext();

function App() {
  const [role, setRole] = useState(null);
  const location = useLocation();
  
  const isLoginPage = location.pathname === '/';
  const isGraphPage = location.pathname === '/graph';

  return (
    <RoleContext.Provider value={{ role, setRole }}>
      <div className="min-h-screen flex flex-col">
        {!isLoginPage && <Navbar />}
        <div className="flex flex-1 overflow-hidden">
          {(!isLoginPage && !isGraphPage) && <Sidebar />}
          <main className={`flex-1 flex flex-col overflow-hidden`}>
            <Routes>
              <Route path="/" element={<LoginPage />} />
              <Route path="/dashboard" element={<CommandCenterPage />} />
              <Route path="/graph" element={<GraphExplorerPage />} />
              <Route path="/alerts/:id" element={<AlertDetailPage />} />
              <Route path="/scenarios" element={<ScenarioControlPage />} />
              <Route path="/cases" element={<CaseExportPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </RoleContext.Provider>
  );
}

export default App;
