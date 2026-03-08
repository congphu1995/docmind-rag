import { create } from "zustand";

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: { email: string; username: string } | null;
  isAuthenticated: boolean;

  login: (token: string, refreshToken: string, user: { email: string; username: string }) => void;
  logout: () => void;
  loadFromStorage: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: null,
  refreshToken: null,
  user: null,
  isAuthenticated: false,

  login: (token, refreshToken, user) => {
    localStorage.setItem("access_token", token);
    localStorage.setItem("refresh_token", refreshToken);
    localStorage.setItem("user", JSON.stringify(user));
    set({ token, refreshToken, user, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("user");
    set({ token: null, refreshToken: null, user: null, isAuthenticated: false });
  },

  loadFromStorage: () => {
    const token = localStorage.getItem("access_token");
    const refreshToken = localStorage.getItem("refresh_token");
    const userStr = localStorage.getItem("user");
    if (token && userStr) {
      set({
        token,
        refreshToken,
        user: JSON.parse(userStr),
        isAuthenticated: true,
      });
    }
  },
}));
