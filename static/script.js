// script.js (Frontend for Chatbot)

document.addEventListener("DOMContentLoaded", () => {
    const chatbotToggler = document.querySelector(".chatbot-toggler");
    const closeBtn = document.querySelector(".close-btn");
    const chatbox = document.querySelector(".chatbox");
    const chatInput = document.querySelector(".chat-input textarea");
    const sendChatBtn = document.querySelector("#send-btn");


    let userMessage = null;
    const inputInitHeight = chatInput.scrollHeight;


    const createChatLi = (message, className) => {
        const chatLi = document.createElement("li");
        chatLi.classList.add("chat", className);
        let chatContent = className === "outgoing"
            ? `<p></p>`
            : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
        chatLi.innerHTML = chatContent;
        
        // Process only bold markdown while preserving original style
        const messageElement = chatLi.querySelector("p");
        // Only convert bold (**text**) to <strong> tags
        const formattedMessage = message.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        messageElement.innerHTML = formattedMessage;
        
        return chatLi;
    };


    const generateResponse = async (chatElement, userMessage) => {
        const messageElement = chatElement.querySelector("p");
        try {
            // Prepare form data
            const formData = new URLSearchParams();
            formData.append('question', userMessage);

            const response = await fetch('/ask_question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: formData.toString(),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();

            if (result.answer) {
                // Only convert bold (**text**) to <strong> tags
                const formattedMessage = result.answer.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                messageElement.innerHTML = formattedMessage;
            } else if (result.error) {
                messageElement.textContent = `Error: ${result.error}`;
            } else {
                messageElement.textContent = "Sorry, I couldn't understand that.";
            }
        } catch (error) {
            messageElement.textContent = "Sorry, something went wrong.";
            console.error('Error:', error);
        }
        chatbox.scrollTo(0, chatbox.scrollHeight);
    };


    const handleChat = () => {
        userMessage = chatInput.value.trim();
        if (!userMessage) return;


        chatInput.value = "";
        chatInput.style.height = `${inputInitHeight}px`;


        chatbox.appendChild(createChatLi(userMessage, "outgoing"));
        chatbox.scrollTo(0, chatbox.scrollHeight);


        setTimeout(() => {
            const incomingChatLi = createChatLi("Thinking...", "incoming");
            chatbox.appendChild(incomingChatLi);
            chatbox.scrollTo(0, chatbox.scrollHeight);
            generateResponse(incomingChatLi, userMessage);
        }, 600);
    };


    chatInput.addEventListener("input", () => {
        chatInput.style.height = `${inputInitHeight}px`;
        chatInput.style.height = `${chatInput.scrollHeight}px`;
    });


    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleChat();
        }
    });


    sendChatBtn.addEventListener("click", handleChat);
    closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
    chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));
});