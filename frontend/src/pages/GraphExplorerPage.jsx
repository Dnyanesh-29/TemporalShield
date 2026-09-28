import { useState, useCallback, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  applyNodeChanges,
  applyEdgeChanges
} from 'reactflow';
import 'reactflow/dist/style.css';
import { nodeTypes } from '../components/graph/CustomNodes';
import { Play, Pause, AlertTriangle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const initialNodes = [
  { id: 'emp-1', type: 'employee', position: { x: 250, y: 150 }, data: { label: 'E-902', role: 'Branch Manager' } },
  { id: 'acc-1', type: 'account', position: { x: 250, y: 300 }, data: { label: 'A-4421', balance: '₹4,50,000' } },
  { id: 'tx-1', type: 'transaction', position: { x: 450, y: 300 }, data: { label: '₹2,00,000', flagged: true, alertId: 'ALT-1042' } },
  { id: 'acc-2', type: 'account', position: { x: 650, y: 300 }, data: { label: 'A-8812' } },
];

const initialEdges = [
  // Access (employee -> account): blue dashed
  { id: 'e1-a1', source: 'emp-1', target: 'acc-1', style: { stroke: '#3b82f6', strokeDasharray: '5,5' } },
  // Suspicious cross-domain: red thick
  { id: 'a1-t1', source: 'acc-1', target: 'tx-1', style: { stroke: '#ef4444', strokeWidth: 3 }, animated: true },
  { id: 't1-a2', source: 'tx-1', target: 'acc-2', style: { stroke: '#ef4444', strokeWidth: 3 }, animated: true },
];

export default function GraphExplorerPage() {
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  const [selectedNode, setSelectedNode] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [timeline, setTimeline] = useState(50);
  const navigate = useNavigate();

  const onNodesChange = useCallback((changes) => setNodes((nds) => applyNodeChanges(changes, nds)), []);
  const onEdgesChange = useCallback((changes) => setEdges((eds) => applyEdgeChanges(changes, eds)), []);

  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  return (
    <div className="relative flex-1 w-full h-full bg-[#0a0a0a]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        className="touch-none"
      >
        <Background color="#333" gap={16} />
        {/* We place native Controls in top-right by default ReactFlow config, but we can customize position */}
        <Controls position="top-right" className="bg-brand-surface border-brand-border fill-gray-300" />
      </ReactFlow>

      {/* NODE LEGEND (Top Left) */}
      <div className="absolute top-4 left-4 bg-brand-surface/90 backdrop-blur border border-brand-border p-3 rounded-lg shadow-xl pointer-events-none">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Graph Legend</h3>
        <div className="flex flex-col gap-2 text-xs text-gray-200">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-600 clip-hexagon border border-blue-400"></div>
            Employee
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-white rounded-full border border-gray-300"></div>
            Account
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-500 rounded border border-gray-400"></div>
            Customer
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 mx-0.5 bg-amber-500 rotate-45 border border-amber-300"></div>
            Transaction
          </div>
          <div className="flex items-center gap-2 mt-1 pt-1 border-t border-brand-border">
            <div className="w-4 h-4 rounded-full bg-red-500/20 border-2 border-red-500 flex items-center justify-center">
               <div className="w-2 h-2 bg-red-500 rounded-full animate-ping"></div>
            </div>
            Flagged Entity
          </div>
        </div>
      </div>

      {/* INLINE TOOLTIP CARD */}
      {selectedNode && (
        <div className="absolute top-4 left-48 ml-8 bg-brand-surface/95 backdrop-blur border border-brand-border p-4 rounded-lg shadow-2xl w-64">
          <div className="flex justify-between items-start mb-2">
            <h4 className="text-white font-bold">{selectedNode.data.label}</h4>
            <span className="text-xs text-gray-400 uppercase">{selectedNode.type}</span>
          </div>
          <div className="text-xs text-gray-300 flex flex-col gap-1 mb-4">
            {Object.entries(selectedNode.data).map(([key, val]) => {
              if (key === 'label' || key === 'flagged' || key === 'alertId') return null;
              return <div key={key}><span className="text-gray-500 capitalize">{key}:</span> {val}</div>;
            })}
          </div>
          
          {selectedNode.data.flagged && selectedNode.data.alertId && (
            <button 
              onClick={() => navigate(`/alerts/${selectedNode.data.alertId}`)}
              className="w-full flex items-center justify-center gap-2 bg-brand-accent/20 hover:bg-brand-accent/40 text-red-400 py-1.5 rounded text-xs font-semibold transition-colors border border-brand-accent/50"
            >
              <AlertTriangle className="w-3 h-3" />
              View Alert Details
            </button>
          )}
        </div>
      )}

      {/* TIMELINE SCRUBBER (Bottom Bar) */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-[90%] max-w-3xl bg-brand-surface/90 backdrop-blur border border-brand-border p-4 rounded-xl shadow-2xl flex items-center gap-4">
        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          className="bg-brand-accent hover:bg-red-700 text-white p-2 rounded-full transition-colors flex-shrink-0"
        >
          {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-0.5" />}
        </button>
        
        <div className="flex-1 flex flex-col gap-1">
          <div className="flex justify-between text-xs text-gray-500 font-mono px-1">
            <span>09:00</span>
            <span>09:15</span>
            <span>09:30</span>
            <span>09:45</span>
            <span>10:00</span>
          </div>
          <input 
            type="range" 
            min="0" 
            max="100" 
            value={timeline}
            onChange={(e) => setTimeline(e.target.value)}
            className="w-full accent-brand-accent cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
}
