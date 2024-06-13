"use strict";

document.addEventListener("DOMContentLoaded", init);

function init() {
    document.querySelector('#submitBtn').addEventListener('click', checkTweet);
}

function checkTweet() {
    const tweetText = document.querySelector('#tweetInput').value;

    // Storing the input to local storage just for testing purpose (can delete if needed)
    localStorage.setItem('tweetText', tweetText);

    fetch('http://your-fastapi-server-ip/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: tweetText })
    })
        .then(function(response) {
            return response.json();
        })
        .then(function(result) {
            const prediction = result.prediction;

            const resultElement = document.querySelector('#result');
            if (prediction === 1) {
                resultElement.textContent = 'Disaster';
                resultElement.style.color = 'red';
            } else {
                resultElement.textContent = 'Non-Disaster';
                resultElement.style.color = 'green';
            }
        })
        .catch(function(error) {
            console.error('Error:', error);
        });
}