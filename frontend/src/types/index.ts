export interface User {
  id: string;
  email: string;
  full_name?: string;
  avatar_url?: string;
  settings?: UserSettings;
}

export interface UserSettings {
  api_keys?: Record<string, string>;
  default_model?: string;
  default_provider?: string;
  custom_endpoint?: string;
  custom_api_key?: string;
  system_prompt?: string;
}

export interface SourceReference {
  source: string;
  page: number | string;
  similarity?: string;
  snippet?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources?: SourceReference[];
  feedback?: 'like' | 'dislike' | null;
  created_at?: string;
}

export interface Conversation {
  id: string;
  title: string;
  system_prompt?: string;
  provider?: string;
  model?: string;
  temperature?: number;
  top_p?: number;
  created_at?: string;
  updated_at?: string;
  message_count?: number;
  messages?: Message[];
}

export interface ModelOption {
  id: string;
  name: string;
  provider: string;
  category?: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  total_chunks: number;
  created_at: string;
}
