import { useState } from "react";
import { useNavigate } from "react-router-dom";
import AuthCard from "@/components/AuthCard";
import { api } from "@/lib/api";

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
      await api.login(username, password);
      localStorage.setItem("isLoggedIn", "true");
      localStorage.setItem("username", username);
      navigate("/dashboard");
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
      await api.register(username, password);
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
