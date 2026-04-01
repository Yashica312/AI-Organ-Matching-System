import { useMemo, useState } from "react";


const API_URL = import.meta.env.VITE_API_URL;

const sidebarItems = ["Dashboard", "Match", "Donors", "Recipients"];
const bloodGroups = ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"];
const organs = ["kidney", "heart", "liver", "lung", "pancreas"];
const urgencyLevels = ["low", "medium", "high", "critical"];

const donorPreview = [
  { id: "D-102", organ: "Kidney", status: "Available" },
  { id: "D-140", organ: "Heart", status: "Screening" },
  { id: "D-155", organ: "Liver", status: "Ready" },
];

const recipientPreview = [
  { id: "R-210", organ: "Kidney", urgency: "High" },
  { id: "R-301", organ: "Heart", urgency: "Critical" },
  { id: "R-344", organ: "Lung", urgency: "Medium" },
];


function App() {
  const [activeSection, setActiveSection] = useState("Match");
  const [formData, setFormData] = useState({
    blood_group: "A+",
    age: 45,
    organ: "kidney",
    urgency: "high",
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const stats = useMemo(
    () => [
      { label: "Active Donors", value: "128" },
      { label: "Waiting Recipients", value: "312" },
      { label: "Successful Matches", value: "87%" },
    ],
    []
  );

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((current) => ({
      ...current,
      [name]: name === "age" ? Number(value) : value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/match`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const payload = await response.json();
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || "Unable to fetch match results.");
    } finally {
      setLoading(false);
    }
  };

  const compatibilityClass =
    result?.compatibility?.toLowerCase() === "high"
      ? "badge badge-high"
      : result?.compatibility?.toLowerCase() === "medium"
        ? "badge badge-medium"
        : "badge badge-low";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">AM</span>
          <div>
            <p className="brand-title">Organ Match</p>
            <p className="brand-subtitle">Clinical dashboard</p>
          </div>
        </div>

        <nav className="nav-list">
          {sidebarItems.map((item) => (
            <button
              key={item}
              type="button"
              className={item === activeSection ? "nav-item active" : "nav-item"}
              onClick={() => setActiveSection(item)}
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>

      <div className="content">
        <header className="header">
          <div>
            <h1>AI Organ Matching System</h1>
            <p>Modern dashboard with ML-backed matching and resilient API fallback.</p>
          </div>
        </header>

        <section className="stats-grid">
          {stats.map((item) => (
            <article key={item.label} className="panel stat-card">
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </article>
          ))}
        </section>

        <section className="main-grid">
          <article className="panel form-panel">
            <div className="panel-header">
              <h2>Match</h2>
              <p>Submit recipient data to find the best organ match.</p>
            </div>

            <form onSubmit={handleSubmit} className="match-form">
              <label>
                Blood Group
                <select name="blood_group" value={formData.blood_group} onChange={handleChange}>
                  {bloodGroups.map((group) => (
                    <option key={group} value={group}>
                      {group}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Age
                <input
                  name="age"
                  type="number"
                  min="0"
                  max="120"
                  value={formData.age}
                  onChange={handleChange}
                />
              </label>

              <label>
                Organ Type
                <select name="organ" value={formData.organ} onChange={handleChange}>
                  {organs.map((organ) => (
                    <option key={organ} value={organ}>
                      {organ}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Urgency
                <select name="urgency" value={formData.urgency} onChange={handleChange}>
                  {urgencyLevels.map((level) => (
                    <option key={level} value={level}>
                      {level}
                    </option>
                  ))}
                </select>
              </label>

              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? "Finding Best Match..." : "Find Best Match"}
              </button>
            </form>

            {loading ? (
              <div className="loading-state">
                <span className="spinner" />
                <p>Processing patient profile and scoring compatibility...</p>
              </div>
            ) : null}

            {error ? <div className="error-banner">{error}</div> : null}
          </article>

          <article className="panel results-panel">
            <div className="panel-header">
              <h2>Results</h2>
              <p>Best available match returned from the API.</p>
            </div>

            {result ? (
              <div className="results-grid">
                <div className="result-card">
                  <span>Donor ID</span>
                  <strong>{result.donor}</strong>
                </div>
                <div className="result-card">
                  <span>Recipient ID</span>
                  <strong>{result.recipient}</strong>
                </div>
                <div className="result-card">
                  <span>Match Score</span>
                  <strong>{result.match_score}%</strong>
                </div>
                <div className="result-card">
                  <span>Compatibility</span>
                  <strong className={compatibilityClass}>{result.compatibility}</strong>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <p>Run a match request to see donor-recipient results here.</p>
              </div>
            )}
          </article>
        </section>

        <section className="main-grid secondary-grid">
          <article className="panel list-panel">
            <div className="panel-header">
              <h2>Donors</h2>
              <p>Quick snapshot of active donor pipeline.</p>
            </div>
            <div className="list">
              {donorPreview.map((donor) => (
                <div key={donor.id} className="list-row">
                  <div>
                    <strong>{donor.id}</strong>
                    <span>{donor.organ}</span>
                  </div>
                  <span className="list-meta">{donor.status}</span>
                </div>
              ))}
            </div>
          </article>

          <article className="panel list-panel">
            <div className="panel-header">
              <h2>Recipients</h2>
              <p>Queue overview by organ and urgency.</p>
            </div>
            <div className="list">
              {recipientPreview.map((recipient) => (
                <div key={recipient.id} className="list-row">
                  <div>
                    <strong>{recipient.id}</strong>
                    <span>{recipient.organ}</span>
                  </div>
                  <span className="list-meta">{recipient.urgency}</span>
                </div>
              ))}
            </div>
          </article>
        </section>
      </div>
    </div>
  );
}


export default App;
