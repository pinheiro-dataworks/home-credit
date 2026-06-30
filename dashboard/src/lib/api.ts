import axios from "axios";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const client = axios.create({ baseURL: BASE, timeout: 15_000 });

export const api = {
  health:           () => client.get("/health"),
  overview:         () => client.get("/api/overview"),
  metrics:          () => client.get("/api/model/metrics"),
  rocCurve:         () => client.get("/api/model/roc"),
  prCurve:          () => client.get("/api/model/pr-curve"),
  calibration:      () => client.get("/api/model/calibration"),
  confusionMatrix:  () => client.get("/api/model/confusion-matrix"),
  featureImportance:(n = 20) => client.get(`/api/features/importance?top_n=${n}`),
  riskDistribution: () => client.get("/api/risk/distribution"),
  statistics:       () => client.get("/api/statistics"),
  sampleApplications:(n = 10) => client.get(`/api/applications/sample?n=${n}`),
  predict:          (data: Record<string, unknown>) => client.post("/api/predict", data),
};

export type { AxiosResponse } from "axios";
