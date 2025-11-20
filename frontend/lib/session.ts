export function getSessionId(): string {
    if (typeof window === 'undefined') return 'server-side-id';
    
    let id = localStorage.getItem('poly_session_id');
    if (!id) {
      id = `user_${crypto.randomUUID().slice(0, 8)}`;
      localStorage.setItem('poly_session_id', id);
    }
    return id;
  }