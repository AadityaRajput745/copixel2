import { useState } from 'react';
import { Container, Row, Col, Form, Button, Card, Accordion } from 'react-bootstrap';

function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  
  const [formStatus, setFormStatus] = useState({
    submitted: false,
    error: false,
    message: ''
  });
  
  const [validated, setValidated] = useState(false);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    
    if (form.checkValidity() === false) {
      e.stopPropagation();
      setValidated(true);
      return;
    }
    
    // Simulate API call
    setTimeout(() => {
      setFormStatus({
        submitted: true,
        error: false,
        message: 'Your message has been sent successfully! We will get back to you soon.'
      });
      
      setFormData({
        name: '',
        email: '',
        subject: '',
        message: ''
      });
      
      setValidated(false);
    }, 1500);
  };
  
  return (
    <>
      <section className="hero-section py-5">
        <Container>
          <Row className="justify-content-center text-center">
            <Col lg={8}>
              <h1 className="display-4 fw-bold">Get In Touch</h1>
              <p className="lead">
                Have questions about our AI detection technology? Want to report suspicious content or join our team? We'd love to hear from you.
              </p>
            </Col>
          </Row>
        </Container>
      </section>
      
      <section className="py-5 bg-pattern">
        <Container>
          <Row>
            <Col lg={6} className="mb-5 mb-lg-0">
              <div className="pe-lg-4 fade-in">
                <div className="badge badge-primary mb-3">Send Us a Message</div>
                <h2 className="fw-bold mb-4">How Can We Help?</h2>
                <p className="mb-4">
                  Fill out the form below to get in touch with our team. We'll respond as soon as possible, typically within 24-48 hours.
                </p>
                
                <Form noValidate validated={validated} onSubmit={handleSubmit} className="contact-form">
                  <Row>
                    <Col md={6} className="mb-3">
                      <Form.Group controlId="contactName">
                        <Form.Label>Your Name</Form.Label>
                        <Form.Control
                          type="text"
                          name="name"
                          value={formData.name}
                          onChange={handleChange}
                          placeholder="Enter your name"
                          required
                        />
                        <Form.Control.Feedback type="invalid">
                          Please provide your name.
                        </Form.Control.Feedback>
                      </Form.Group>
                    </Col>
                    
                    <Col md={6} className="mb-3">
                      <Form.Group controlId="contactEmail">
                        <Form.Label>Email Address</Form.Label>
                        <Form.Control
                          type="email"
                          name="email"
                          value={formData.email}
                          onChange={handleChange}
                          placeholder="Enter your email"
                          required
                        />
                        <Form.Control.Feedback type="invalid">
                          Please provide a valid email address.
                        </Form.Control.Feedback>
                      </Form.Group>
                    </Col>
                  </Row>
                  
                  <Form.Group className="mb-3" controlId="contactSubject">
                    <Form.Label>Subject</Form.Label>
                    <Form.Control
                      type="text"
                      name="subject"
                      value={formData.subject}
                      onChange={handleChange}
                      placeholder="What is this regarding?"
                      required
                    />
                    <Form.Control.Feedback type="invalid">
                      Please provide a subject.
                    </Form.Control.Feedback>
                  </Form.Group>
                  
                  <Form.Group className="mb-4" controlId="contactMessage">
                    <Form.Label>Message</Form.Label>
                    <Form.Control
                      as="textarea"
                      name="message"
                      value={formData.message}
                      onChange={handleChange}
                      placeholder="How can we help you?"
                      rows={5}
                      required
                    />
                    <Form.Control.Feedback type="invalid">
                      Please provide a message.
                    </Form.Control.Feedback>
                  </Form.Group>
                  
                  <div className="d-grid">
                    <Button 
                      variant="primary" 
                      type="submit" 
                      size="lg"
                      className="rounded-pill"
                      disabled={formStatus.submitted}
                    >
                      {formStatus.submitted ? (
                        <>
                          <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                          Sending...
                        </>
                      ) : (
                        <>
                          <i className="fas fa-paper-plane me-2"></i> Send Message
                        </>
                      )}
                    </Button>
                  </div>
                  
                  {formStatus.submitted && !formStatus.error && (
                    <div className="alert alert-success mt-4 animate__animated animate__fadeIn" role="alert">
                      <i className="fas fa-check-circle me-2"></i>
                      {formStatus.message}
                    </div>
                  )}
                  
                  {formStatus.submitted && formStatus.error && (
                    <div className="alert alert-danger mt-4 animate__animated animate__fadeIn" role="alert">
                      <i className="fas fa-exclamation-circle me-2"></i>
                      {formStatus.message}
                    </div>
                  )}
                </Form>
              </div>
            </Col>
            
            <Col lg={6}>
              <div className="slide-in">
                <div className="badge badge-primary mb-3">Contact Information</div>
                <h2 className="fw-bold mb-4">Reach Out to Us</h2>
                
                <Row className="mb-4">
                  {[
                    {
                      icon: 'fa-map-marker-alt',
                      title: 'Visit Our Office',
                      content: '123 AI Innovation Center, Tech Park,<br />San Francisco, CA 94103',
                      link: 'https://maps.google.com',
                      linkText: 'Get Directions'
                    },
                    {
                      icon: 'fa-envelope',
                      title: 'Email Us',
                      content: 'contact@copixel.tech<br />support@copixel.tech',
                      link: 'mailto:contact@copixel.tech',
                      linkText: 'Send Email'
                    },
                    {
                      icon: 'fa-phone-alt',
                      title: 'Call Us',
                      content: '+1 (555) 123-4567<br />Mon-Fri, 9:00 AM - 6:00 PM PST',
                      link: 'tel:+15551234567',
                      linkText: 'Call Now'
                    }
                  ].map((item, index) => (
                    <Col md={6} className="mb-4" key={index}>
                      <Card className="h-100 border-0 shadow-sm contact-card">
                        <Card.Body className="p-4">
                          <div className="icon-wrapper mb-3">
                            <i className={`fas ${item.icon}`}></i>
                          </div>
                          <Card.Title className="fw-bold">{item.title}</Card.Title>
                          <Card.Text dangerouslySetInnerHTML={{ __html: item.content }}></Card.Text>
                          <a href={item.link} className="btn-link" target="_blank" rel="noopener noreferrer">
                            {item.linkText} <i className="fas fa-arrow-right ms-1"></i>
                          </a>
                        </Card.Body>
                      </Card>
                    </Col>
                  ))}
                </Row>
                
                <Card className="border-0 shadow-sm mb-5 report-card">
                  <Card.Body className="p-4">
                    <div className="d-flex align-items-center mb-3">
                      <div className="icon-wrapper me-3 bg-danger">
                        <i className="fas fa-exclamation-triangle"></i>
                      </div>
                      <h4 className="fw-bold mb-0">Report Abuse</h4>
                    </div>
                    <Card.Text>
                      Have you found AI-generated content being used for harmful purposes? Report it to our content abuse team for immediate action.
                    </Card.Text>
                    <a href="/report" className="btn btn-danger rounded-pill">
                      <i className="fas fa-flag me-2"></i> Submit a Report
                    </a>
                  </Card.Body>
                </Card>
                
                <div className="social-media-links d-flex justify-content-start">
                  {[
                    { platform: 'twitter', url: '#', icon: 'fa-twitter' },
                    { platform: 'facebook', url: '#', icon: 'fa-facebook-f' },
                    { platform: 'linkedin', url: '#', icon: 'fa-linkedin-in' },
                    { platform: 'github', url: '#', icon: 'fa-github' }
                  ].map((social, index) => (
                    <a 
                      key={index} 
                      href={social.url} 
                      className={`social-icon ${social.platform}`}
                      target="_blank" 
                      rel="noopener noreferrer"
                      aria-label={`Follow us on ${social.platform}`}
                    >
                      <i className={`fab ${social.icon}`}></i>
                    </a>
                  ))}
                </div>
              </div>
            </Col>
          </Row>
        </Container>
      </section>
      
      <section className="py-5 bg-light">
        <Container>
          <div className="text-center mb-5 fade-in">
            <div className="badge badge-primary mb-3">FAQ</div>
            <h2 className="fw-bold mb-4">Frequently Asked Questions</h2>
            <p className="text-muted mx-auto" style={{ maxWidth: '700px' }}>
              Find answers to commonly asked questions about our AI detection system and services
            </p>
          </div>
          
          <Row className="justify-content-center">
            <Col lg={8}>
              <Accordion defaultActiveKey="0" className="faq-accordion fade-in">
                {[
                  {
                    question: 'How accurate is your AI detection system?',
                    answer: 'Our AI detection system achieves an average accuracy of 99.5% for deepfake videos, 98.7% for document forgery, and 99.2% for signature verification. We continuously train and improve our models to maintain high accuracy as new AI generation techniques emerge.'
                  },
                  {
                    question: 'What file formats do you support for analysis?',
                    answer: 'We support a wide range of file formats. For video analysis, we accept MP4, MOV, and AVI formats. For document forgery detection, we accept PDF, JPEG, and PNG files. For signature verification, we accept JPEG, PNG, and TIFF images.'
                  },
                  {
                    question: 'How long does it take to analyze content?',
                    answer: 'Most analyses are completed within seconds. Video deepfake detection typically takes 2-5 seconds per video, document analysis takes 1-3 seconds per page, and signature verification takes under 2 seconds. Processing time may vary based on file size and complexity.'
                  },
                  {
                    question: 'Is my data secure when I use your service?',
                    answer: 'Yes, we take data security very seriously. All uploads are encrypted during transit and storage. Your files are automatically deleted from our servers after analysis unless you specifically request retention for reporting purposes. We never share your data with third parties.'
                  },
                  {
                    question: 'How do I report suspected AI-generated content?',
                    answer: 'You can use our reporting system by clicking the "Submit a Report" button on our contact page or after receiving detection results. Provide as much information as possible about the content and where you found it. Our team will review the report and take appropriate action.'
                  },
                  {
                    question: 'Do you offer an API for integration with our systems?',
                    answer: 'Yes, we offer a comprehensive API that allows you to integrate our AI detection capabilities into your own applications and workflows. Contact our sales team for API documentation, pricing, and integration support.'
                  }
                ].map((faq, index) => (
                  <Accordion.Item eventKey={index.toString()} key={index}>
                    <Accordion.Header>
                      <span className="fw-bold">{faq.question}</span>
                    </Accordion.Header>
                    <Accordion.Body>
                      {faq.answer}
                    </Accordion.Body>
                  </Accordion.Item>
                ))}
              </Accordion>
              
              <div className="text-center mt-5">
                <p className="mb-4">Still have questions? Contact our support team directly.</p>
                <a href="mailto:support@copixel.tech" className="btn btn-primary rounded-pill px-4 py-2">
                  <i className="fas fa-envelope me-2"></i> Email Support
                </a>
              </div>
            </Col>
          </Row>
        </Container>
      </section>
      
      <style jsx="true">{`
        .contact-form .form-control {
          padding: 0.75rem 1.25rem;
          border-radius: 10px;
          border: 1px solid rgba(0, 0, 0, 0.1);
          background-color: #f8f9fa;
          transition: all 0.3s ease;
        }
        
        .contact-form .form-control:focus {
          background-color: #fff;
          box-shadow: 0 0 0 0.25rem rgba(var(--primary-rgb), 0.25);
          border-color: var(--primary-color);
        }
        
        .contact-card {
          border-radius: 10px;
          overflow: hidden;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .contact-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1) !important;
        }
        
        .btn-link {
          color: var(--primary-color);
          text-decoration: none;
          font-weight: 600;
          transition: color 0.3s ease;
          display: inline-flex;
          align-items: center;
        }
        
        .btn-link:hover {
          color: var(--secondary-color);
        }
        
        .btn-link i {
          transition: transform 0.3s ease;
        }
        
        .btn-link:hover i {
          transform: translateX(5px);
        }
        
        .report-card {
          background: linear-gradient(to right, #fff, #f8f9fa);
          border-radius: 10px;
        }
        
        .social-media-links {
          gap: 15px;
        }
        
        .social-icon {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #fff;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .social-icon:hover {
          transform: translateY(-5px);
          box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
        }
        
        .social-icon.twitter {
          background-color: #1DA1F2;
        }
        
        .social-icon.facebook {
          background-color: #4267B2;
        }
        
        .social-icon.linkedin {
          background-color: #0A66C2;
        }
        
        .social-icon.github {
          background-color: #333;
        }
        
        .faq-accordion .accordion-item {
          border: none;
          margin-bottom: 1rem;
          border-radius: 10px;
          overflow: hidden;
          box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        }
        
        .faq-accordion .accordion-button {
          padding: 1.25rem;
          background-color: #fff;
          box-shadow: none;
        }
        
        .faq-accordion .accordion-button:not(.collapsed) {
          color: var(--primary-color);
          background-color: #fff;
          box-shadow: none;
        }
        
        .faq-accordion .accordion-button:focus {
          box-shadow: none;
          border-color: rgba(0, 0, 0, 0.125);
        }
        
        .faq-accordion .accordion-body {
          padding: 1rem 1.25rem 1.5rem;
          background-color: #fff;
        }
      `}</style>
    </>
  );
}

export default ContactPage;

 