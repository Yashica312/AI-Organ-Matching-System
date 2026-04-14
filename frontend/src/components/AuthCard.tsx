import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Eye, EyeOff, Heart, LogIn, UserPlus } from "lucide-react";

interface AuthCardProps {
  onLogin?: (username: string, password: string) => Promise<void> | void;
  onSignup?: (username: string, password: string) => Promise<void> | void;
  loading?: boolean;
  loginErrorMessage?: string;
  signupErrorMessage?: string;
  signupSuccessMessage?: string;
}

const AuthCard = ({
  onLogin,
  onSignup,
  loading = false,
  loginErrorMessage = "",
  signupErrorMessage = "",
  signupSuccessMessage = "",
}: AuthCardProps) => {
  const [activeTab, setActiveTab] = useState("login");

  // Login state
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [showLoginPassword, setShowLoginPassword] = useState(false);
  const [loginError, setLoginError] = useState("");

  // Signup state
  const [signupUsername, setSignupUsername] = useState("");
  const [signupPassword, setSignupPassword] = useState("");
  const [signupConfirm, setSignupConfirm] = useState("");
  const [showSignupPassword, setShowSignupPassword] = useState(false);
  const [signupError, setSignupError] = useState("");
  const [signupSuccess, setSignupSuccess] = useState("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError("");

    if (!loginUsername.trim() || !loginPassword.trim()) {
      setLoginError("Please fill in all fields.");
      return;
    }

    if (onLogin) {
      await onLogin(loginUsername, loginPassword);
    } else {
      if (loginUsername === "admin" && loginPassword === "password") {
        setLoginError("");
        alert("Login successful!");
      } else {
        setLoginError("Invalid username or password.");
      }
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setSignupError("");
    setSignupSuccess("");

    if (!signupUsername.trim() || !signupPassword.trim() || !signupConfirm.trim()) {
      setSignupError("Please fill in all fields.");
      return;
    }

    if (signupPassword.length < 6) {
      setSignupError("Password must be at least 6 characters.");
      return;
    }

    if (signupPassword !== signupConfirm) {
      setSignupError("Passwords do not match.");
      return;
    }

    if (onSignup) {
      await onSignup(signupUsername, signupPassword);
    } else {
      setSignupSuccess("Account created successfully! You can now log in.");
      setSignupUsername("");
      setSignupPassword("");
      setSignupConfirm("");
    }
  };

  return (
    <div className="w-full max-w-[450px] mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 mb-4">
          <Heart className="w-7 h-7 text-primary" />
        </div>
        <h1 className="text-2xl font-semibold text-foreground tracking-tight">
          AI Organ Matcher
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Intelligent donor-recipient matching system
        </p>
      </div>

      {/* Auth Card */}
      <div
        className="bg-card rounded-2xl border border-border p-8"
        style={{ boxShadow: "var(--shadow-card)" }}
      >
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2 mb-6 bg-muted rounded-xl h-11">
            <TabsTrigger
              value="login"
              className="rounded-lg text-sm font-medium data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-sm"
            >
              <LogIn className="w-4 h-4 mr-2" />
              Sign In
            </TabsTrigger>
            <TabsTrigger
              value="signup"
              className="rounded-lg text-sm font-medium data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-sm"
            >
              <UserPlus className="w-4 h-4 mr-2" />
              Sign Up
            </TabsTrigger>
          </TabsList>

          {/* Login Tab */}
          <TabsContent value="login">
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="login-username" className="text-sm font-medium text-foreground">
                  Username
                </Label>
                <Input
                  id="login-username"
                  placeholder="Enter your username"
                  value={loginUsername}
                  onChange={(e) => setLoginUsername(e.target.value)}
                  className="h-11 rounded-xl border-border bg-muted/50 placeholder:text-muted-foreground/60"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="login-password" className="text-sm font-medium text-foreground">
                  Password
                </Label>
                <div className="relative">
                  <Input
                    id="login-password"
                    type={showLoginPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    className="h-11 rounded-xl border-border bg-muted/50 pr-10 placeholder:text-muted-foreground/60"
                  />
                  <button
                    type="button"
                    onClick={() => setShowLoginPassword(!showLoginPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showLoginPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {(loginError || loginErrorMessage) && (
                <div className="text-sm text-destructive bg-destructive/10 rounded-lg px-3 py-2">
                  {loginError || loginErrorMessage}
                </div>
              )}

              <Button disabled={loading} type="submit" className="w-full h-11 rounded-xl text-sm font-medium mt-2">
                {loading ? "Signing In..." : "Sign In"}
              </Button>
            </form>
          </TabsContent>

          {/* Signup Tab */}
          <TabsContent value="signup">
            <form onSubmit={handleSignup} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="signup-username" className="text-sm font-medium text-foreground">
                  Username
                </Label>
                <Input
                  id="signup-username"
                  placeholder="Choose a username"
                  value={signupUsername}
                  onChange={(e) => setSignupUsername(e.target.value)}
                  className="h-11 rounded-xl border-border bg-muted/50 placeholder:text-muted-foreground/60"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="signup-password" className="text-sm font-medium text-foreground">
                  Password
                </Label>
                <div className="relative">
                  <Input
                    id="signup-password"
                    type={showSignupPassword ? "text" : "password"}
                    placeholder="Create a password"
                    value={signupPassword}
                    onChange={(e) => setSignupPassword(e.target.value)}
                    className="h-11 rounded-xl border-border bg-muted/50 pr-10 placeholder:text-muted-foreground/60"
                  />
                  <button
                    type="button"
                    onClick={() => setShowSignupPassword(!showSignupPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showSignupPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="signup-confirm" className="text-sm font-medium text-foreground">
                  Confirm Password
                </Label>
                <Input
                  id="signup-confirm"
                  type={showSignupPassword ? "text" : "password"}
                  placeholder="Confirm your password"
                  value={signupConfirm}
                  onChange={(e) => setSignupConfirm(e.target.value)}
                  className="h-11 rounded-xl border-border bg-muted/50 placeholder:text-muted-foreground/60"
                />
              </div>

              {(signupError || signupErrorMessage) && (
                <div className="text-sm text-destructive bg-destructive/10 rounded-lg px-3 py-2">
                  {signupError || signupErrorMessage}
                </div>
              )}

              {(signupSuccess || signupSuccessMessage) && (
                <div className="text-sm text-success bg-success/10 rounded-lg px-3 py-2">
                  {signupSuccess || signupSuccessMessage}
                </div>
              )}

              <Button disabled={loading} type="submit" className="w-full h-11 rounded-xl text-sm font-medium mt-2">
                {loading ? "Creating Account..." : "Create Account"}
              </Button>
            </form>
          </TabsContent>
        </Tabs>
      </div>

      <p className="text-center text-xs text-muted-foreground mt-6">
        Secure authentication for medical professionals
      </p>
    </div>
  );
};

export default AuthCard;
