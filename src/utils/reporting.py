import os
import json
import logging
import uuid
from datetime import datetime
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import dotenv

# Load environment variables
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

class ReportingSystem:
    def __init__(self):
        # Default authorities contact information
        # In a real implementation, this would come from a secure database
        self.authorities = {
            "cybercrime": {
                "name": "Cyber Crime Division",
                "email": os.getenv("CYBER_CRIME_EMAIL", "cybercrime@example.gov"),
                "api_endpoint": os.getenv("CYBER_CRIME_API", "https://api.cybercrime.example.gov/report"),
                "api_key": os.getenv("CYBER_CRIME_API_KEY", "")
            },
            "document_fraud": {
                "name": "Document Fraud Investigation Unit",
                "email": os.getenv("DOC_FRAUD_EMAIL", "document.fraud@example.gov"),
                "api_endpoint": os.getenv("DOC_FRAUD_API", "https://api.documentfraud.example.gov/report"),
                "api_key": os.getenv("DOC_FRAUD_API_KEY", "")
            },
            "general": {
                "name": "Fraud Reporting Center",
                "email": os.getenv("GENERAL_FRAUD_EMAIL", "fraud.report@example.gov"),
                "api_endpoint": os.getenv("GENERAL_FRAUD_API", "https://api.fraudreport.example.gov/report"),
                "api_key": os.getenv("GENERAL_FRAUD_API_KEY", "")
            }
        }
        
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.example.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "ai.detection@example.com")
        
        # Results directory
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       '..', '..', 'data', 'results')
    
    def get_detection_result(self, detection_id):
        """Retrieve a detection result by ID"""
        try:
            result_path = os.path.join(self.results_dir, f"{detection_id}.json")
            if not os.path.exists(result_path):
                logger.error(f"Detection result not found: {detection_id}")
                return None
                
            with open(result_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving detection result: {str(e)}")
            return None
    
    def determine_authority(self, detection_result):
        """Determine which authority to report to based on detection type"""
        # Extract file info to determine content type
        filename = detection_result.get("filename", "")
        file_extension = os.path.splitext(filename)[1].lower() if filename else ""
        
        # Check content type from details
        details = detection_result.get("details", {})
        
        if "faces_detected" in details:
            # This is likely a video deepfake
            return "cybercrime"
        elif "text_analysis" in details:
            # This is likely a document forgery
            return "document_fraud"
        elif "feature_analysis" in details:
            # This is likely a signature forgery
            return "document_fraud"
        else:
            # Default to general reporting
            return "general"
    
    def prepare_report_data(self, detection_result, report_data):
        """Prepare data for reporting to authorities"""
        # Generate a unique report ID
        report_id = str(uuid.uuid4())
        
        # Combine detection result with additional report information
        report = {
            "report_id": report_id,
            "timestamp": datetime.now().isoformat(),
            "detection_result": detection_result,
            "reporter_info": {
                "name": report_data.get("reporter_name", "Anonymous"),
                "contact": report_data.get("reporter_contact", ""),
                "organization": report_data.get("reporter_organization", ""),
                "notes": report_data.get("notes", "")
            },
            "evidence_location": report_data.get("evidence_location", "")
        }
        
        return report
    
    def send_email_report(self, authority, report):
        """Send a report via email"""
        if not self.smtp_username or not self.smtp_password:
            logger.warning("SMTP credentials not configured, email report not sent")
            return False
            
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = authority["email"]
            msg['Subject'] = f"AI-Generated Content Detection Report {report['report_id']}"
            
            # Add text body
            body = f"""
            AI-Generated Content Detection Report
            
            Report ID: {report['report_id']}
            Submitted: {report['timestamp']}
            
            Detection Details:
            - Content Type: {os.path.splitext(report['detection_result'].get('filename', ''))[1]}
            - Filename: {report['detection_result'].get('filename', 'Not specified')}
            - Detection ID: {report['detection_result'].get('detection_id', 'Not specified')}
            - AI Generated: {'Yes' if report['detection_result'].get('is_fake', False) else 'No'}
            - Confidence: {report['detection_result'].get('confidence', 0) * 100:.1f}%
            
            Reporter Information:
            - Name: {report['reporter_info'].get('name', 'Anonymous')}
            - Contact: {report['reporter_info'].get('contact', 'Not provided')}
            - Organization: {report['reporter_info'].get('organization', 'Not provided')}
            
            Additional Notes:
            {report['reporter_info'].get('notes', 'None provided')}
            
            Please review the attached JSON file for complete detection results.
            """
            msg.attach(MIMEText(body, 'plain'))
            
            # Add JSON attachment
            json_attachment = MIMEApplication(json.dumps(report, indent=2), _subtype="json")
            json_attachment.add_header('Content-Disposition', 'attachment', filename=f"report_{report['report_id']}.json")
            msg.attach(json_attachment)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                
            logger.info(f"Email report sent to {authority['name']} ({authority['email']})")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email report: {str(e)}")
            return False
    
    def send_api_report(self, authority, report):
        """Send a report via API"""
        if not authority["api_key"]:
            logger.warning(f"API key not configured for {authority['name']}, API report not sent")
            return False
            
        try:
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {authority['api_key']}",
                "X-Report-Source": "AI-Detection-System"
            }
            
            # Send request
            response = requests.post(
                authority["api_endpoint"],
                headers=headers,
                json=report,
                timeout=30
            )
            
            # Check response
            if response.status_code == 200:
                logger.info(f"API report successfully sent to {authority['name']}")
                return True
            else:
                logger.error(f"API report failed with status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending API report: {str(e)}")
            return False
    
    def save_report_locally(self, report):
        """Save the report locally"""
        try:
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     '..', '..', 'data', 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Save report
            report_path = os.path.join(reports_dir, f"report_{report['report_id']}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Report saved locally to {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving report locally: {str(e)}")
            return False
    
    def report(self, report_data):
        """Main reporting method"""
        logger.info(f"Processing report for detection: {report_data.get('detection_id', 'unknown')}")
        
        # Get the detection result
        detection_result = self.get_detection_result(report_data.get("detection_id"))
        if not detection_result:
            return {
                "success": False,
                "error": "Detection result not found"
            }
        
        # Determine which authority to report to
        authority_key = report_data.get("authority") or self.determine_authority(detection_result)
        authority = self.authorities.get(authority_key, self.authorities["general"])
        
        # Prepare report data
        report = self.prepare_report_data(detection_result, report_data)
        
        # Save report locally
        local_save_success = self.save_report_locally(report)
        
        # Try API report first
        api_success = False
        if authority["api_endpoint"] and authority["api_key"]:
            api_success = self.send_api_report(authority, report)
        
        # Fall back to email if API fails or not configured
        email_success = False
        if (not api_success) and authority["email"]:
            email_success = self.send_email_report(authority, report)
        
        result = {
            "success": api_success or email_success or local_save_success,
            "report_id": report["report_id"],
            "reported_to": authority["name"],
            "reporting_methods": {
                "api": api_success,
                "email": email_success,
                "local_save": local_save_success
            }
        }
        
        return result 