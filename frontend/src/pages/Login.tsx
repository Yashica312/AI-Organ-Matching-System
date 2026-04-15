import { useState } from "react";
import { useNavigate } from "react-router-dom";
import AuthCard from "@/components/AuthCard";

const Login = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [signupError, setSignupError] = useState("");
  const [signupSuccess, setSignupSuccess] = useState("");

  const handleLogin = async (username: string, password: string) => {
    setLoading(true);
    setLoginError("");
    setSignupError("");
    setSignupSuccess("");
    try {
      const res = await fetch("https://ai-organ-matching-system.onrender.com/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) throw new Error("Invalid credentials");
      localStorage.setItem("isLoggedIn", "true");
      localStorage.setItem("username", username);
      navigate("/matching");
    } catch {
      setLoginError("Invalid credentials");
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (username: string, password: string) => {
    setLoading(true);
    setLoginError("");
    setSignupError("");
    setSignupSuccess("");
    try {
      const baseUrl = import.meta.env.VITE_API_URL || "https://ai-organ-matching-system.onrender.com";
      const res = await fetch(`${baseUrl}/api/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) throw new Error("Registration failed");
      setSignupSuccess("Account created. Please log in.");
    } catch {
      setSignupError("Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4 py-12">
      <AuthCard
        onLogin={handleLogin}
        onSignup={handleSignup}
        loading={loading}
        loginErrorMessage={loginError}
        signupErrorMessage={signupError}
        signupSuccessMessage={signupSuccess}
      />
    </div>
  );
};

export default Login;
