document.addEventListener('DOMContentLoaded', function() {
    // Form submission handlers
    setupFormHandler('videoForm', handleDetectionResponse);
    setupFormHandler('documentForm', handleDetectionResponse);
    setupFormHandler('signatureForm', handleDetectionResponse);
    
    // Report form handler
    const reportForm = document.getElementById('reportForm');
    if (reportForm) {
        reportForm.addEventListener('submit', handleReportSubmission);
    }
    
    // Contact form handler
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            alert('Thank you for your message. We will get back to you soon!');
            contactForm.reset();
        });
    }
});

/**
 * Sets up AJAX form submission
 * @param {string} formId - ID of the form to handle
 * @param {function} callback - Callback function on successful response
 */
function setupFormHandler(formId, callback) {
    const form = document.getElementById(formId);
    if (!form) return;
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading state
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.textContent = 'Processing...';
        submitBtn.disabled = true;
        
        // Get form data
        const formData = new FormData(form);
        
        // Send AJAX request
        fetch(form.action, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Reset form UI
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
            
            // Handle response
            if (callback) callback(data);
        })
        .catch(error => {
            console.error('Error:', error);
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
            showErrorMessage('An error occurred during processing. Please try again.');
        });
    });
}

/**
 * Handles detection API response
 * @param {object} data - Response data from detection API
 */
function handleDetectionResponse(data) {
    // Show results section
    const resultsSection = document.getElementById('results');
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    const resultContent = document.getElementById('resultContent');
    const reportSection = document.getElementById('reportSection');
    
    // Clear previous results
    resultContent.innerHTML = '';
    
    if (data.success) {
        // Build result HTML
        const resultHTML = createResultHTML(data);
        resultContent.innerHTML = resultHTML;
        
        // Setup confidence meter
        setupConfidenceMeter(data.confidence);
        
        // Show report section if fake detected
        if (data.is_fake) {
            reportSection.style.display = 'block';
            document.getElementById('detectionId').value = data.detection_id;
        } else {
            reportSection.style.display = 'none';
        }
    } else {
        // Show error message
        resultContent.innerHTML = `
            <div class="alert alert-danger">
                <h4>Error</h4>
                <p>${data.error || 'An error occurred during processing'}</p>
            </div>
        `;
        reportSection.style.display = 'none';
    }
}

/**
 * Creates HTML for detection results
 * @param {object} data - Detection result data
 * @returns {string} HTML string
 */
function createResultHTML(data) {
    const isFake = data.is_fake;
    const confidence = data.confidence;
    const confidencePercent = Math.round(confidence * 100);
    const statusClass = isFake ? 'is-fake' : 'not-fake';
    const statusText = isFake ? 'AI-Generated Content Detected' : 'Likely Authentic Content';
    const statusColor = isFake ? 'danger' : 'success';
    
    let html = `
        <div class="result-panel ${statusClass} fade-in">
            <h3 class="text-${statusColor} mb-3">${statusText}</h3>
            <p><strong>File:</strong> ${data.filename || 'Uploaded file'}</p>
            <p><strong>Confidence:</strong> ${confidencePercent}%</p>
            
            <div class="confidence-meter">
                <div class="meter-fill" style="width: ${confidencePercent}%"></div>
            </div>
    `;
    
    // Add details based on the content type
    if (data.details) {
        html += '<div class="mt-4"><h4>Analysis Details</h4>';
        
        if (data.details.faces_detected !== undefined) {
            // Video analysis
            html += `
                <p><strong>Frames Analyzed:</strong> ${data.details.frames_processed || 'N/A'}</p>
                <p><strong>Faces Detected:</strong> ${data.details.faces_detected || 0}</p>
            `;
            
            if (data.details.inconsistencies) {
                html += '<p><strong>Detected Issues:</strong> ';
                
                const issues = [];
                const inconsistencies = data.details.inconsistencies;
                
                if (inconsistencies.unrealistic_features) {
                    issues.push('Unrealistic facial features');
                }
                if (inconsistencies.temporal_inconsistencies) {
                    issues.push('Temporal inconsistencies between frames');
                }
                
                html += issues.length ? issues.join(', ') : 'None detected';
                html += '</p>';
            }
        } else if (data.details.text_analysis) {
            // Document analysis
            html += `<p><strong>Document Type:</strong> ${getFileType(data.filename)}</p>`;
            
            const textAnalysis = data.details.text_analysis;
            const visualAnalysis = data.details.visual_analysis || {};
            
            if (textAnalysis.suspicious_patterns) {
                html += `
                    <p class="text-danger"><strong>AI-Generated Text Patterns Detected</strong></p>
                    <p><strong>Suspicious Phrases:</strong> ${textAnalysis.suspicious_words?.join(', ') || 'None specifically identified'}</p>
                `;
            }
            
            if (visualAnalysis.digital_artifacts) {
                html += `<p class="text-danger"><strong>Digital Manipulation Artifacts Detected</strong></p>`;
            }
            
            if (visualAnalysis.signature_anomalies) {
                html += `<p class="text-danger"><strong>Signature Anomalies Detected</strong></p>`;
            }
        } else if (data.details.feature_analysis) {
            // Signature analysis
            html += `<p><strong>Signature Analysis:</strong></p>`;
            
            const consistency = data.details.consistency_analysis || {};
            
            if (consistency.unusual_stroke_pattern) {
                html += `<p class="text-danger"><strong>Unusual Stroke Patterns Detected</strong></p>`;
            }
            
            if (consistency.tremor_signs) {
                html += `<p class="text-danger"><strong>Tremor Signs Detected</strong> (possible manual forgery)</p>`;
            }
            
            if (consistency.ai_generation_signs) {
                html += `<p class="text-danger"><strong>AI Generation Markers Detected</strong></p>`;
            }
        }
        
        html += '</div>';
    }
    
    // Add detection ID (hidden in UI but useful for debugging)
    html += `<div class="mt-4 pt-3 border-top text-muted small">
                <p>Detection ID: ${data.detection_id}</p>
                <p>Detection completed: ${formatDate(data.timestamp)}</p>
            </div>`;
    
    html += '</div>';
    
    if (isFake) {
        html += `
            <div class="alert alert-warning fade-in mt-3">
                <h5>Important Notice</h5>
                <p>This content has been identified as likely AI-generated or manipulated. If you believe this content is being used for harmful purposes, please use the reporting form below to notify relevant authorities.</p>
            </div>
        `;
    }
    
    return html;
}

/**
 * Handles report form submission
 * @param {Event} e - Submit event
 */
function handleReportSubmission(e) {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    
    submitBtn.textContent = 'Submitting...';
    submitBtn.disabled = true;
    
    // Collect form data as JSON
    const formData = new FormData(form);
    const jsonData = {};
    
    formData.forEach((value, key) => {
        jsonData[key] = value;
    });
    
    // Send report
    fetch('/api/report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(jsonData)
    })
    .then(response => response.json())
    .then(data => {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
        
        if (data.success) {
            // Show success message
            const reportSection = document.getElementById('reportSection');
            reportSection.innerHTML = `
                <div class="alert alert-success mt-4">
                    <h4>Report Submitted Successfully</h4>
                    <p>Your report has been submitted to ${data.reported_to}.</p>
                    <p>Report ID: ${data.report_id}</p>
                    <p>Thank you for helping to combat the misuse of AI-generated content.</p>
                </div>
            `;
        } else {
            // Show error message
            showErrorMessage(data.error || 'An error occurred while submitting the report. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
        showErrorMessage('An error occurred while submitting the report. Please try again.');
    });
}

/**
 * Displays an error message
 * @param {string} message - Error message to display
 */
function showErrorMessage(message) {
    const resultContent = document.getElementById('resultContent');
    
    const errorHTML = `
        <div class="alert alert-danger fade-in">
            <h4>Error</h4>
            <p>${message}</p>
        </div>
    `;
    
    resultContent.innerHTML = errorHTML;
}

/**
 * Sets up the confidence meter
 * @param {number} confidence - Confidence value (0-1)
 */
function setupConfidenceMeter(confidence) {
    const meterFill = document.querySelector('.meter-fill');
    if (!meterFill) return;
    
    const confidencePercent = Math.round(confidence * 100);
    
    // Animate the confidence meter
    setTimeout(() => {
        meterFill.style.width = `${confidencePercent}%`;
    }, 100);
}

/**
 * Gets human-readable file type from filename
 * @param {string} filename - Filename with extension
 * @returns {string} Human-readable file type
 */
function getFileType(filename) {
    if (!filename) return 'Unknown';
    
    const extension = filename.split('.').pop().toLowerCase();
    
    const types = {
        'pdf': 'PDF Document',
        'jpg': 'JPEG Image',
        'jpeg': 'JPEG Image',
        'png': 'PNG Image',
        'mp4': 'MP4 Video',
        'avi': 'AVI Video',
        'mov': 'QuickTime Video',
        'mkv': 'MKV Video'
    };
    
    return types[extension] || extension.toUpperCase();
}

/**
 * Formats a date string
 * @param {string} dateString - ISO date string
 * @returns {string} Formatted date string
 */
function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch (e) {
        return dateString;
    }
} 