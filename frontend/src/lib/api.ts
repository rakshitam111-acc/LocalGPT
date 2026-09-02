const API_BASE = '/api';

export function getAuthToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('localgpt_token');
  }
  return null;
}

export function setAuthToken(token: string) {
  if (typeof window !== 'undefined') {
    localStorage.setItem('localgpt_token', token);
  }
}

export function removeAuthToken() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('localgpt_token');
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorMsg = 'An error occurred';
    try {
      const errJson = await response.json();
      errorMsg = errJson.detail || errJson.message || errorMsg;
    } catch {
      errorMsg = response.statusText;
    }
    throw new Error(errorMsg);
  }

  return response.json();
}

export const api = {
  // Auth
  login: (data: any) => request<any>('/auth/login', { method: 'POST', body: JSON.stringify(data) }),
  register: (data: any) => request<any>('/auth/register', { method: 'POST', body: JSON.stringify(data) }),
  googleAuth: (data: any) => request<any>('/auth/google', { method: 'POST', body: JSON.stringify(data) }),
  getMe: () => request<any>('/auth/me'),
  updateSettings: (data: any) => request<any>('/auth/settings', { method: 'PATCH', body: JSON.stringify(data) }),
  resetPassword: (data: any) => request<any>('/auth/reset-password', { method: 'POST', body: JSON.stringify(data) }),

  // Conversations
  getConversations: (search?: string) => request<any[]>(`/conversations${search ? `?search=${encodeURIComponent(search)}` : ''}`),
  createConversation: (data: any) => request<any>('/conversations', { method: 'POST', body: JSON.stringify(data) }),
  getConversation: (id: string) => request<any>(`/conversations/${id}`),
  updateConversation: (id: string, data: any) => request<any>(`/conversations/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  deleteConversation: (id: string) => request<any>(`/conversations/${id}`, { method: 'DELETE' }),
  setMessageFeedback: (msgId: string, feedback: string | null) => request<any>(`/messages/${msgId}/feedback`, { method: 'POST', body: JSON.stringify({ feedback }) }),

  // Documents
  uploadDocuments: (formData: FormData) => request<any>('/documents/upload', { method: 'POST', body: formData }),
  getDocuments: () => request<any[]>('/documents'),
  deleteDocument: (id: string) => request<any>(`/documents/${id}`, { method: 'DELETE' }),
  clearDocuments: () => request<any>('/documents', { method: 'DELETE' }),

  // Models
  getModels: () => request<any>('/models'),
};
