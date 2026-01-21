import React, { useState, useEffect, useRef } from 'react';
import { initializeApp } from 'firebase/app';
import { 
  getAuth, 
  signInWithCustomToken, 
  signInAnonymously, 
  onAuthStateChanged 
} from 'firebase/auth';
import { 
  getFirestore, 
  collection, 
  doc, 
  setDoc, 
  getDoc, 
  onSnapshot,
  query,
  addDoc,
  updateDoc
} from 'firebase/firestore';
import { 
  Send, 
  User, 
  Bot, 
  Image as ImageIcon, 
  Loader2, 
  RefreshCw,
  Search,
  Volume2
} from 'lucide-react';

// --- Firebase Configuration & Initialization ---
const firebaseConfig = JSON.parse(__firebase_config);
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-gemini-app';
const apiKey = ""; // API Key is provided by the environment

const App = () => {
  const [user, setUser] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);

  // --- Authentication Logic (Rule 3) ---
  useEffect(() => {
    const initAuth = async () => {
      try {
        if (typeof __initial_auth_token !== 'undefined' && __initial_auth_token) {
          await signInWithCustomToken(auth, __initial_auth_token);
        } else {
          await signInAnonymously(auth);
        }
      } catch (err) {
        console.error("Auth error:", err);
        setError("身份驗證失敗，請重新整理頁面。");
      }
    };
    initAuth();
    const unsubscribe = onAuthStateChanged(auth, setUser);
    return () => unsubscribe();
  }, []);

  // --- Firestore Data Fetching (Rule 1 & 2) ---
  useEffect(() => {
    if (!user) return;

    const q = collection(db, 'artifacts', appId, 'users', user.uid, 'messages');
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const msgs = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      // Sorting in memory per Rule 2
      setMessages(msgs.sort((a, b) => a.timestamp - b.timestamp));
    }, (err) => {
      console.error("Firestore error:", err);
    });

    return () => unsubscribe();
  }, [user]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // --- Gemini API Call with Exponential Backoff ---
  const callGemini = async (prompt, retryCount = 0) => {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key=${apiKey}`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }]
        })
      });

      if (!response.ok) {
        if (response.status === 429 && retryCount < 5) {
          const delay = Math.pow(2, retryCount) * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
          return callGemini(prompt, retryCount + 1);
        }
        throw new Error('API 請求失敗');
      }

      const data = await response.json();
      return data.candidates?.[0]?.content?.parts?.[0]?.text || "無回應內容";
    } catch (err) {
      if (retryCount < 5) {
        const delay = Math.pow(2, retryCount) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
        return callGemini(prompt, retryCount + 1);
      }
      throw err;
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || !user || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      // Save User Message
      const userMsgRef = collection(db, 'artifacts', appId, 'users', user.uid, 'messages');
      await addDoc(userMsgRef, {
        text: userMessage,
        role: 'user',
        timestamp: Date.now()
      });

      // Get AI Response
      const aiResponse = await callGemini(userMessage);

      // Save AI Message
      await addDoc(userMsgRef, {
        text: aiResponse,
        role: 'bot',
        timestamp: Date.now()
      });
    } catch (err) {
      setError("發生錯誤，請稍後再試。");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Header */}
      <header className="bg-white border-b px-6 py-4 flex justify-between items-center shadow-sm">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Bot size={20} className="text-white" />
          </div>
          <h1 className="font-bold text-xl tracking-tight">Gemini 3 Flash</h1>
        </div>
        <div className="text-xs text-slate-400 bg-slate-100 px-3 py-1 rounded-full uppercase tracking-widest">
          {user ? `UID: ${user.uid}` : '正在連線...'}
        </div>
      </header>

      {/* Chat Area */}
      <main 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6"
      >
        {messages.length === 0 && !isLoading && (
          <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-4">
            <div className="p-4 bg-white rounded-2xl shadow-sm border border-slate-100">
              <Bot size={48} className="text-slate-200" />
            </div>
            <p className="text-sm">輸入訊息以開始對話</p>
          </div>
        )}

        {messages.map((msg) => (
          <div 
            key={msg.id} 
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex gap-3 max-w-[85%] md:max-w-[70%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center ${
                msg.role === 'user' ? 'bg-slate-200' : 'bg-blue-100'
              }`}>
                {msg.role === 'user' ? <User size={16} /> : <Bot size={16} className="text-blue-600" />}
              </div>
              <div className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-tr-none' 
                  : 'bg-white text-slate-800 border border-slate-100 rounded-tl-none'
              }`}>
                {msg.text}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="flex gap-3 items-center text-slate-400">
              <div className="w-8 h-8 rounded-full bg-blue-50 flex items-center justify-center">
                <Loader2 size={16} className="animate-spin text-blue-400" />
              </div>
              <span className="text-xs animate-pulse">思考中...</span>
            </div>
          </div>
        )}
        
        {error && (
          <div className="bg-red-50 text-red-600 p-3 rounded-lg text-xs text-center border border-red-100">
            {error}
          </div>
        )}
      </main>

      {/* Input Area */}
      <footer className="p-4 bg-white border-t">
        <form 
          onSubmit={handleSend}
          className="max-w-4xl mx-auto flex gap-2 items-center bg-slate-100 p-2 rounded-2xl focus-within:ring-2 ring-blue-500/20 transition-all"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="輸入您的問題..."
            className="flex-1 bg-transparent border-none focus:outline-none px-4 py-2 text-sm"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 text-white p-2.5 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors shadow-lg shadow-blue-500/30"
          >
            <Send size={18} />
          </button>
        </form>
        <p className="text-[10px] text-center text-slate-400 mt-3">
          此應用程式使用 Gemini 3 Flash Preview 模型進行測試
        </p>
      </footer>
    </div>
  );
};

export default App;
