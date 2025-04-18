// static/js/script.js

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const inputText = document.getElementById('input-text');
    const stemmerSelect = document.getElementById('stemmer-select');
    const stemBtn = document.getElementById('stem-btn');
    const compareBtn = document.getElementById('compare-btn');
    const resultsSection = document.getElementById('results-section');
    const comparisonBody = document.getElementById('comparison-body');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const searchQuery = document.getElementById('search-query');
    const searchStemmer = document.getElementById('search-stemmer');
    const searchBtn = document.getElementById('search-btn');
    const searchResultsContainer = document.getElementById('search-results-container');
    
    // Chart objects
    let vocabularyChart, timeChart, lengthChart, accuracyChart;
    let porterPRChart, snowballPRChart, lancasterPRChart;
    
    // State variables
    let hasComparedText = false;
    
    // Event listeners
    stemBtn.addEventListener('click', handleStemming);
    compareBtn.addEventListener('click', handleComparison);
    searchBtn.addEventListener('click', handleSearch);
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });
    
    // Function to handle stemming with a single stemmer
    function handleStemming() {
        const text = inputText.value.trim();
        const stemmer = stemmerSelect.value;
        
        if (!text) {
            alert('Please enter some text to stem');
            return;
        }
        
        fetch('/stem', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text, stemmer })
        })
        .then(response => response.json())
        .then(data => {
            // Show results section
            resultsSection.classList.remove('hidden');
            
            // Create table for stemming results
            let tableContent = '';
            data.stemmed_tokens.forEach(item => {
                tableContent += `
                    <tr>
                        <td>${item.original}</td>
                        <td>${item.stemmed}</td>
                    </tr>
                `;
            });
            
            comparisonBody.innerHTML = tableContent;
            
            // Activate first tab
            tabBtns[0].click();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while stemming the text');
        });
    }
    
    // Function to handle comparison of all stemmers
    function handleComparison() {
        const text = inputText.value.trim();
        
        if (!text) {
            alert('Please enter some text to compare');
            return;
        }
        
        fetch('/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Unknown error occurred');
                });
            }
            return response.json();
        })
        .then(data => {
            // Show results section
            resultsSection.classList.remove('hidden');
            hasComparedText = true;
            
            // Create table for comparison results
            let tableContent = '';
            data.comparison.forEach(item => {
                // Highlight differences
                const porterDiff = item.porter !== item.snowball || item.porter !== item.lancaster;
                const snowballDiff = item.snowball !== item.porter || item.snowball !== item.lancaster;
                const lancasterDiff = item.lancaster !== item.porter || item.lancaster !== item.snowball;
                
                tableContent += `
                    <tr>
                        <td>${item.original}</td>
                        <td class="${porterDiff ? 'different' : ''}">${item.porter}</td>
                        <td class="${snowballDiff ? 'different' : ''}">${item.snowball}</td>
                        <td class="${lancasterDiff ? 'different' : ''}">${item.lancaster}</td>
                    </tr>
                `;
            });
            
            comparisonBody.innerHTML = tableContent;
            
            // Update metrics
            document.getElementById('porter-accuracy').textContent = `${(data.evaluation.porter.accuracy * 100).toFixed(2)}%`;
            document.getElementById('snowball-accuracy').textContent = `${(data.evaluation.snowball.accuracy * 100).toFixed(2)}%`;
            document.getElementById('lancaster-accuracy').textContent = `${(data.evaluation.lancaster.accuracy * 100).toFixed(2)}%`;
            
            document.getElementById('porter-reduction').textContent = `${data.vocabulary_reduction.porter.reduction_percentage.toFixed(2)}%`;
            document.getElementById('snowball-reduction').textContent = `${data.vocabulary_reduction.snowball.reduction_percentage.toFixed(2)}%`;
            document.getElementById('lancaster-reduction').textContent = `${data.vocabulary_reduction.lancaster.reduction_percentage.toFixed(2)}%`;
            
            document.getElementById('porter-time').textContent = `${(data.processing_time.porter * 1000).toFixed(3)} ms`;
            document.getElementById('snowball-time').textContent = `${(data.processing_time.snowball * 1000).toFixed(3)} ms`;
            document.getElementById('lancaster-time').textContent = `${(data.processing_time.lancaster * 1000).toFixed(3)} ms`;
            
            // Update precision, recall, and F1 score metrics
            if (data.metrics) {
                document.getElementById('porter-precision').textContent = `${(data.metrics.porter.precision * 100).toFixed(2)}%`;
                document.getElementById('porter-recall').textContent = `${(data.metrics.porter.recall * 100).toFixed(2)}%`;
                document.getElementById('porter-f1').textContent = `${(data.metrics.porter.f1 * 100).toFixed(2)}%`;
                document.getElementById('porter-overstemming').textContent = data.metrics.porter.overstemming;
                document.getElementById('porter-understemming').textContent = data.metrics.porter.understemming;
                
                document.getElementById('snowball-precision').textContent = `${(data.metrics.snowball.precision * 100).toFixed(2)}%`;
                document.getElementById('snowball-recall').textContent = `${(data.metrics.snowball.recall * 100).toFixed(2)}%`;
                document.getElementById('snowball-f1').textContent = `${(data.metrics.snowball.f1 * 100).toFixed(2)}%`;
                document.getElementById('snowball-overstemming').textContent = data.metrics.snowball.overstemming;
                document.getElementById('snowball-understemming').textContent = data.metrics.snowball.understemming;
                
                document.getElementById('lancaster-precision').textContent = `${(data.metrics.lancaster.precision * 100).toFixed(2)}%`;
                document.getElementById('lancaster-recall').textContent = `${(data.metrics.lancaster.recall * 100).toFixed(2)}%`;
                document.getElementById('lancaster-f1').textContent = `${(data.metrics.lancaster.f1 * 100).toFixed(2)}%`;
                document.getElementById('lancaster-overstemming').textContent = data.metrics.lancaster.overstemming;
                document.getElementById('lancaster-understemming').textContent = data.metrics.lancaster.understemming;
                
                // Create or update precision-recall charts
                createPRCharts(data.metrics);
            }
            
            // Create or update charts
            createVocabularyChart(data.vocabulary_reduction);
            createTimeChart(data.processing_time);
            createAccuracyChart(data.evaluation);
            createLengthChart(data.comparison);
            
            // Activate first tab
            tabBtns[0].click();
        })
        .catch(error => {
            console.error('Error:', error);
            alert(error.message || 'An error occurred while comparing stemmers');
        });
    }
    
    // Function to create precision-recall charts
    function createPRCharts(metrics) {
        // Porter PR Chart
        const porterCtx = document.getElementById('porter-pr-chart').getContext('2d');
        if (porterPRChart) porterPRChart.destroy();
        porterPRChart = createSinglePRChart(porterCtx, metrics.porter, 'Porter');
        
        // Snowball PR Chart
        const snowballCtx = document.getElementById('snowball-pr-chart').getContext('2d');
        if (snowballPRChart) snowballPRChart.destroy();
        snowballPRChart = createSinglePRChart(snowballCtx, metrics.snowball, 'Snowball');
        
        // Lancaster PR Chart
        const lancasterCtx = document.getElementById('lancaster-pr-chart').getContext('2d');
        if (lancasterPRChart) lancasterPRChart.destroy();
        lancasterPRChart = createSinglePRChart(lancasterCtx, metrics.lancaster, 'Lancaster');
    }
    
    // Helper function to create a single PR chart
    function createSinglePRChart(ctx, metrics, stemmerName) {
        return new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Precision', 'Recall', 'F1 Score'],
                datasets: [{
                    label: stemmerName,
                    data: [
                        metrics.precision * 100,
                        metrics.recall * 100,
                        metrics.f1 * 100
                    ],
                    backgroundColor: 'rgba(79, 195, 161, 0.2)',
                    borderColor: 'rgba(79, 195, 161, 1)',
                    pointBackgroundColor: 'rgba(79, 195, 161, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(79, 195, 161, 1)'
                }]
            },
            options: {
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0,
                        suggestedMax: 100,
                        ticks: {
                            stepSize: 20
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // Function to handle search testing
    function handleSearch() {
        const query = searchQuery.value.trim();
        const stemmer = searchStemmer.value;
        
        if (!query) {
            alert('Please enter a search query');
            return;
        }
        
        if (!hasComparedText) {
            searchResultsContainer.innerHTML = `
                <div class="search-info-message">
                    <p>Please first use the "Compare All Stemmers" feature to provide text for searching.</p>
                    <p>The search will look for terms within the text you analyzed.</p>
                </div>
            `;
            return;
        }
        
        fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query, stemmer })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Unknown error occurred');
                });
            }
            return response.json();
        })
        .then(data => {
            // Show results
            if (data.results.length === 0) {
                searchResultsContainer.innerHTML = '<p class="no-results">No results found in your text. Try a different query or stemmer.</p>';
            } else {
                let resultsHTML = '';
                data.results.forEach(result => {
                    resultsHTML += `
                        <div class="search-result-item">
                            <div class="search-score">Score: ${result.score}</div>
                            <div class="search-document">${result.document}</div>
                            ${result.note ? `<div class="search-note">${result.note}</div>` : ''}
                        </div>
                    `;
                });
                searchResultsContainer.innerHTML = resultsHTML;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            searchResultsContainer.innerHTML = `<p class="search-error">${error.message || 'An error occurred during search'}</p>`;
        });
    }
    
    // Function to create vocabulary reduction chart
    function createVocabularyChart(data) {
        const ctx = document.getElementById('vocabulary-chart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (vocabularyChart) {
            vocabularyChart.destroy();
        }
        
        vocabularyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Porter', 'Snowball', 'Lancaster'],
                datasets: [
                    {
                        label: 'Vocabulary Reduction (%)',
                        data: [
                            data.porter.reduction_percentage,
                            data.snowball.reduction_percentage,
                            data.lancaster.reduction_percentage
                        ],
                        backgroundColor: [
                            'rgba(74, 111, 165, 0.7)',
                            'rgba(22, 96, 136, 0.7)',
                            'rgba(79, 195, 161, 0.7)'
                        ],
                        borderColor: [
                            'rgba(74, 111, 165, 1)',
                            'rgba(22, 96, 136, 1)',
                            'rgba(79, 195, 161, 1)'
                        ],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Reduction (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // Function to create processing time chart
    function createTimeChart(data) {
        const ctx = document.getElementById('time-chart').getContext('2d');
        
        // Convert to milliseconds for display
        const porterTime = data.porter * 1000;
        const snowballTime = data.snowball * 1000;
        const lancasterTime = data.lancaster * 1000;
        
        // Destroy existing chart if it exists
        if (timeChart) {
            timeChart.destroy();
        }
        
        timeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Porter', 'Snowball', 'Lancaster'],
                datasets: [
                    {
                        label: 'Processing Time (ms)',
                        data: [porterTime, snowballTime, lancasterTime],
                        backgroundColor: [
                            'rgba(74, 111, 165, 0.7)',
                            'rgba(22, 96, 136, 0.7)',
                            'rgba(79, 195, 161, 0.7)'
                        ],
                        borderColor: [
                            'rgba(74, 111, 165, 1)',
                            'rgba(22, 96, 136, 1)',
                            'rgba(79, 195, 161, 1)'
                        ],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (milliseconds)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // Function to create accuracy chart
    function createAccuracyChart(data) {
        const ctx = document.getElementById('accuracy-chart').getContext('2d');
        
        // Convert to percentage for display
        const porterAccuracy = data.porter.accuracy * 100;
        const snowballAccuracy = data.snowball.accuracy * 100;
        const lancasterAccuracy = data.lancaster.accuracy * 100;
        
        // Destroy existing chart if it exists
        if (accuracyChart) {
            accuracyChart.destroy();
        }
        
        accuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Porter', 'Snowball', 'Lancaster'],
                datasets: [
                    {
                        label: 'Accuracy (%)',
                        data: [porterAccuracy, snowballAccuracy, lancasterAccuracy],
                        backgroundColor: [
                            'rgba(74, 111, 165, 0.7)',
                            'rgba(22, 96, 136, 0.7)',
                            'rgba(79, 195, 161, 0.7)'
                        ],
                        borderColor: [
                            'rgba(74, 111, 165, 1)',
                            'rgba(22, 96, 136, 1)',
                            'rgba(79, 195, 161, 1)'
                        ],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // Function to create length reduction chart
    function createLengthChart(comparisonData) {
        const ctx = document.getElementById('length-chart').getContext('2d');
        
        // Calculate average length reduction
        let porterTotalReduction = 0;
        let snowballTotalReduction = 0;
        let lancasterTotalReduction = 0;
        let count = 0;
        
        comparisonData.forEach(item => {
            const originalLength = item.original.length;
            porterTotalReduction += originalLength - item.porter.length;
            snowballTotalReduction += originalLength - item.snowball.length;
            lancasterTotalReduction += originalLength - item.lancaster.length;
            count++;
        });
        
        const porterAvg = count > 0 ? porterTotalReduction / count : 0;
        const snowballAvg = count > 0 ? snowballTotalReduction / count : 0;
        const lancasterAvg = count > 0 ? lancasterTotalReduction / count : 0;
        
        // Destroy existing chart if it exists
        if (lengthChart) {
            lengthChart.destroy();
        }
        
        lengthChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Porter', 'Snowball', 'Lancaster'],
                datasets: [
                    {
                        label: 'Average Character Reduction',
                        data: [porterAvg, snowballAvg, lancasterAvg],
                        backgroundColor: [
                            'rgba(74, 111, 165, 0.7)',
                            'rgba(22, 96, 136, 0.7)',
                            'rgba(79, 195, 161, 0.7)'
                        ],
                        borderColor: [
                            'rgba(74, 111, 165, 1)',
                            'rgba(22, 96, 136, 1)',
                            'rgba(79, 195, 161, 1)'
                        ],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Average Characters Removed'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
});