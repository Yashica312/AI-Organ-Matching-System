import { useNavigate } from "react-router-dom";
import AuthCard from "@/components/AuthCard";

const Login = () => {
  const navigate = useNavigate();

  const handleLogin = (username: string, password: string) => {
    // Demo: accept any non-empty credentials
    if (username && password) {
      navigate("/dashboard");
    }
  };

  const handleSignup = (username: string, password: string) => {
    // Demo: auto-navigate after signup
    navigate("/dashboard");
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4 py-12">
      <AuthCard onLogin={handleLogin} onSignup={handleSignup} />
    </div>
  );
};

export default Login;
