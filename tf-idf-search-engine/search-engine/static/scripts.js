function sendData() {
  // Get the input value
  var searchInput = document.getElementById("search-input").value;

  // Create a FormData object
  var formData = new FormData();
  formData.append("search", searchInput);

  // Make a POST request using the Fetch API
  fetch("http://localhost:5000//result", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      // Handle the response from the server
      console.log(data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
