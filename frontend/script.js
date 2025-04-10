document.getElementById("searchBtn").addEventListener("click", function() {
  const query = document.getElementById("queryInput").value;
  if (!query) return;

  fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query: query })
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById("result").innerText = "Matched Fund: " + data.matched_fund;
  })
  .catch(error => {
    console.error("Error:", error);
    document.getElementById("result").innerText = "An error occurred.";
  });
});
