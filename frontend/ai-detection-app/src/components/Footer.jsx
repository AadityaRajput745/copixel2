import { Link } from 'react-router-dom';
import { Container, Row, Col } from 'react-bootstrap';

function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="footer-waves">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" preserveAspectRatio="none">
          <path
            fill="var(--primary-color)"
            fillOpacity="0.1"
            d="M0,288L48,272C96,256,192,224,288,197.3C384,171,480,149,576,165.3C672,181,768,235,864,250.7C960,267,1056,245,1152,224C1248,203,1344,181,1392,170.7L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
          ></path>
          <path
            fill="var(--primary-color)"
            fillOpacity="0.2"
            d="M0,224L48,213.3C96,203,192,181,288,154.7C384,128,480,96,576,122.7C672,149,768,235,864,261.3C960,288,1056,256,1152,229.3C1248,203,1344,181,1392,170.7L1440,160L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
          ></path>
        </svg>
      </div>

      <div className="footer-main py-5">
        <Container>
          <Row className="mb-5">
            <Col lg={4} md={6} className="mb-4 mb-lg-0">
              <div className="footer-brand mb-4">
                <img src="/logo.svg" alt="COPixel" width="150" className="mb-3" />
                <p>
                  Advanced AI detection technology to identify deepfakes, 
                  document forgeries, and AI-generated content with 
                  industry-leading accuracy.
                </p>
              </div>
              
              <div className="social-links">
                {[
                  { icon: 'twitter', url: '#' },
                  { icon: 'facebook-f', url: '#' },
                  { icon: 'linkedin-in', url: '#' },
                  { icon: 'github', url: '#' }
                ].map((social, index) => (
                  <a 
                    key={index} 
                    href={social.url} 
                    className="social-icon"
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    <i className={`fab fa-${social.icon}`}></i>
                  </a>
                ))}
              </div>
            </Col>
            
            <Col lg={2} md={6} className="mb-4 mb-lg-0">
              <h5 className="footer-heading">Company</h5>
              <ul className="footer-links">
                <li><Link to="/about">About Us</Link></li>
                <li><Link to="/contact">Contact</Link></li>
                <li><Link to="/careers">Careers</Link></li>
                <li><Link to="/partners">Partners</Link></li>
                <li><Link to="/press">Press Kit</Link></li>
              </ul>
            </Col>
            
            <Col lg={2} md={6} className="mb-4 mb-lg-0">
              <h5 className="footer-heading">Features</h5>
              <ul className="footer-links">
                <li><Link to="/">Video Deepfake</Link></li>
                <li><Link to="/">Document Forgery</Link></li>
                <li><Link to="/">Signature Forgery</Link></li>
                <li><Link to="/api-docs">API Access</Link></li>
                <li><Link to="/enterprise">Enterprise</Link></li>
              </ul>
            </Col>
            
            <Col lg={2} md={6} className="mb-4 mb-lg-0">
              <h5 className="footer-heading">Resources</h5>
              <ul className="footer-links">
                <li><Link to="/blog">Blog</Link></li>
                <li><Link to="/documentation">Documentation</Link></li>
                <li><Link to="/tutorials">Tutorials</Link></li>
                <li><Link to="/research">Research</Link></li>
                <li><Link to="/faq">FAQ</Link></li>
              </ul>
            </Col>
            
            <Col lg={2} md={6}>
              <h5 className="footer-heading">Legal</h5>
              <ul className="footer-links">
                <li><Link to="/terms">Terms of Service</Link></li>
                <li><Link to="/privacy">Privacy Policy</Link></li>
                <li><Link to="/cookies">Cookie Policy</Link></li>
                <li><Link to="/compliance">Compliance</Link></li>
                <li><Link to="/security">Security</Link></li>
              </ul>
            </Col>
          </Row>
          
          <hr className="footer-divider" />
          
          <div className="footer-bottom d-flex flex-wrap justify-content-between align-items-center">
            <p className="copyright mb-0">
              &copy; {currentYear} COPixel. All rights reserved.
            </p>
            
            <div className="d-flex align-items-center">
              <select className="language-selector me-3">
                <option value="en">English</option>
                <option value="es">Español</option>
                <option value="fr">Français</option>
                <option value="de">Deutsch</option>
              </select>
              
              <div className="theme-toggle">
                <button className="theme-btn" title="Toggle dark mode">
                  <i className="fas fa-moon"></i>
                </button>
              </div>
            </div>
          </div>
        </Container>
      </div>
      
      <style jsx="true">{`
        .footer {
          position: relative;
          margin-top: 6rem;
        }
        
        .footer-waves {
          position: absolute;
          top: -120px;
          left: 0;
          width: 100%;
          height: 120px;
          overflow: hidden;
        }
        
        .footer-waves svg {
          width: 100%;
          height: 100%;
        }
        
        .footer-main {
          background-color: var(--dark-color);
          color: rgba(255, 255, 255, 0.8);
          position: relative;
        }
        
        .footer-brand p {
          opacity: 0.8;
          font-size: 0.95rem;
        }
        
        .social-links {
          display: flex;
          gap: 12px;
        }
        
        .social-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 38px;
          height: 38px;
          border-radius: 50%;
          background-color: rgba(255, 255, 255, 0.1);
          color: #fff;
          transition: all 0.3s ease;
        }
        
        .social-icon:hover {
          background-color: var(--primary-color);
          transform: translateY(-3px);
        }
        
        .footer-heading {
          color: #fff;
          font-weight: 600;
          margin-bottom: 1.5rem;
          position: relative;
          padding-bottom: 10px;
        }
        
        .footer-heading::after {
          content: '';
          position: absolute;
          left: 0;
          bottom: 0;
          width: 30px;
          height: 2px;
          background-color: var(--primary-color);
        }
        
        .footer-links {
          list-style: none;
          padding: 0;
          margin: 0;
        }
        
        .footer-links li {
          margin-bottom: 0.8rem;
        }
        
        .footer-links a {
          color: rgba(255, 255, 255, 0.7);
          text-decoration: none;
          transition: all 0.3s ease;
          position: relative;
          padding-left: 0;
        }
        
        .footer-links a:hover {
          color: var(--primary-color);
          padding-left: 5px;
        }
        
        .footer-divider {
          border-color: rgba(255, 255, 255, 0.1);
          margin: 1.5rem 0;
        }
        
        .copyright {
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.9rem;
        }
        
        .language-selector {
          background-color: rgba(255, 255, 255, 0.1);
          border: none;
          color: rgba(255, 255, 255, 0.8);
          padding: 0.4rem 1rem;
          border-radius: 5px;
          font-size: 0.9rem;
          outline: none;
        }
        
        .theme-btn {
          background-color: rgba(255, 255, 255, 0.1);
          border: none;
          color: rgba(255, 255, 255, 0.8);
          width: 38px;
          height: 38px;
          border-radius: 5px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all 0.3s ease;
        }
        
        .theme-btn:hover {
          background-color: rgba(255, 255, 255, 0.2);
          color: #fff;
        }
        
        @media (max-width: 768px) {
          .footer-bottom {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
          }
          
          .footer-heading {
            margin-top: 1.5rem;
          }
          
          .footer-heading::after {
            left: 50%;
            transform: translateX(-50%);
          }
          
          .footer-links {
            text-align: center;
          }
        }
      `}</style>
    </footer>
  );
}

export default Footer; 