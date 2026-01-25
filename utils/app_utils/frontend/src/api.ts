import type { ExampleImage, ProcessingResponse } from './types';

const BASE = '';  // same-origin; Vite proxy handles /api in dev

export async function fetchExamples(): Promise<ExampleImage[]> {
  const res = await fetch(`${BASE}/api/examples`);
  if (!res.ok) throw new Error('Failed to fetch examples');
  return res.json();
}

export async function processImage(file: File): Promise<ProcessingResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/api/process`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Processing failed');
  }
  return res.json();
}

export async function processExample(name: string): Promise<ProcessingResponse> {
  const res = await fetch(`${BASE}/api/process?example=${encodeURIComponent(name)}`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Processing failed');
  }
  return res.json();
}

export async function fetchCachedResult(name: string): Promise<ProcessingResponse> {
  const res = await fetch(`${BASE}/api/cached-result/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error('No cached result');
  return res.json();
}

export function exampleImageUrl(name: string): string {
  return `${BASE}/api/example-image/${encodeURIComponent(name)}`;
}

export function floorplanImageUrl(name: string): string {
  return `${BASE}/api/floorplan-image/${encodeURIComponent(name)}`;
}
