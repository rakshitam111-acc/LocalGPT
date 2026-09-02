import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function groupConversationsByDate<T extends { updated_at?: string; created_at?: string }>(items: T[]): Record<string, T[]> {
  const groups: Record<string, T[]> = {
    'Today': [],
    'Yesterday': [],
    'Previous 7 Days': [],
    'Previous 30 Days': [],
    'Older': [],
  };

  const now = new Date();
  const oneDay = 24 * 60 * 60 * 1000;

  items.forEach(item => {
    const dateStr = item.updated_at || item.created_at;
    if (!dateStr) {
      groups['Older'].push(item);
      return;
    }

    const itemDate = new Date(dateStr);
    const diffDays = Math.floor((now.getTime() - itemDate.getTime()) / oneDay);

    if (diffDays === 0) {
      groups['Today'].push(item);
    } else if (diffDays === 1) {
      groups['Yesterday'].push(item);
    } else if (diffDays <= 7) {
      groups['Previous 7 Days'].push(item);
    } else if (diffDays <= 30) {
      groups['Previous 30 Days'].push(item);
    } else {
      groups['Older'].push(item);
    }
  });

  return Object.fromEntries(Object.entries(groups).filter(([_, list]) => list.length > 0));
}
