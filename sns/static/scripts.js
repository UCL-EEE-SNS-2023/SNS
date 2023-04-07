document.addEventListener("DOMContentLoaded", function () {
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const chatContent = document.getElementById("chat-content");

    // Show welcome messages from chatbot
    displayMessage("Hello, I am the Oracle! I will provide you with predictive data for Tesla's stock in the coming days.", "received");

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });


    function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            // Show messages sent by users
            displayMessage(message, "sent");

            // Clear input box
            userInput.value = "";

            // Sending messages to the back end and receiving replies
            fetch("http://127.0.0.1:5000/message", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: message }),
            })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error("Network response was not ok");
                    }
                    return response.json();
                })
                .then((data) => {
                    // Show incoming messages
                    displayMessage(data.reply, "received", data.image_url);
                })
                .catch((error) => {
                    console.error("There was a problem with the fetch operation:", error);
                });
        }
    }

    function displayMessage(text, messageType, imageUrl) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", messageType);
        messageDiv.textContent = text;


        if (imageUrl) {
            const image = document.createElement("img");
            image.src = imageUrl;
            image.classList.add("message-image");
            messageDiv.appendChild(image);
        }


        chatContent.appendChild(messageDiv);

        // Scroll to Latest News
        chatContent.scrollTop = chatContent.scrollHeight;
    }
});
