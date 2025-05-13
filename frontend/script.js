async function checkFakeNews() {
  const inputText = document.getElementById("inputText").value;
  console.log("🔍 Sending text to API:", inputText);

  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText })
    });
    console.log("📥 Got HTTP status:", res.status);

    const data = await res.json();
    console.log("📊 Response JSON:", data);

    document.getElementById("result").innerText = "Prediction: " + data.result;
  } catch (err) {
    console.error("❌ Fetch error:", err);
    document.getElementById("result").innerText = "Error: " + err.message;
  }
}
