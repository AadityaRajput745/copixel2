import { useState, useRef } from 'react';
import { Container, Row, Col, Tabs, Tab, Alert, Button } from 'react-bootstrap';
import DetectionForm from '../components/DetectionForm';
import ResultPanel from '../components/ResultPanel';

// Import icons for detector cards
const VideoIcon = () => <i className="fas fa-video fa-2x text-white"></i>;
const DocumentIcon = () => <i className="fas fa-file-alt fa-2x text-white"></i>;
const SignatureIcon = () => <i className="fas fa-signature fa-2x text-white"></i>;

function HomePage() {
  const [results, setResults] = useState([]);
  const detectionRef = useRef(null);
  
  const handleDetectionSubmit = (result) => {
    setResults(prevResults => [result, ...prevResults]);
  };

  const scrollToDetection = () => {
    detectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  return (
    <>
      {/* Hero Section */}
      <section className="hero-section py-5">
        <Container>
          <Row className="align-items-center">
            <Col lg={6} className="mb-5 mb-lg-0">
              <h1 className="display-4 fw-bold">Detect AI-Generated Content with Confidence</h1>
              <p className="lead mb-4">
                Our advanced AI detection technology identifies deepfakes, forged documents, and fake signatures with industry-leading accuracy.
              </p>
              <div className="d-flex flex-wrap gap-3">
                <a href="#" onClick={(e) => { e.preventDefault(); scrollToDetection(); }} className="cta-button">
                  <i className="fas fa-shield-alt me-2"></i> Start Detection
                </a>
                <a href="#" className="cta-button">
                  <i className="fas fa-shield-alt me-2"></i> Learn More
                </a>
              </div>
            </Col>
            <Col lg={6}>
              <div className="position-relative">
                {/* <img 
                  src="https://via.placeholder.com/600x400?text=AI+Detection"
                  alt="AI Detection System" 
                  className="img-fluid rounded-lg shadow-lg float-animation"
                /> */}
                <div className="position-absolute top-0 start-100 translate-middle badge badge-primary p-3 shadow-sm">
                  <i className="fas fa-shield-alt me-1"></i> 90% + Accurate
                </div>
                {/* <div className="position-absolute top-100 start-0 translate-middle badge badge-success p-3 shadow-sm">
                  <i className="fas fa-bolt me-1"></i> Fast Results
                </div> */}
              </div>
            </Col>
          </Row>
        </Container>
      </section>
      
      {/* Stats Section */}
      <section className="py-4 bg-pattern">
        <Container>
          <Row className="text-center">
            {[
              { icon: 'fa-shield-alt', number: '90% +', text: 'Detection Accuracy' },
              { icon: 'fa-bolt', number: '2.5s', text: 'Average Process Time' },
              { icon: 'fa-users', number: '50k+', text: 'Active Users' },
              { icon: 'fa-file-shield', number: '1M+', text: 'Files Analyzed' }
            ].map((stat, index) => (
              <Col key={index} md={3} sm={6} className="mb-4 mb-md-0">
                <div className="p-3">
                  <div className="mb-3">
                    <span className="icon-wrapper d-inline-flex">
                      <i className={`fas ${stat.icon}`}></i>
                    </span>
                  </div>
                  <h2 className="fw-bold mb-1">{stat.number}</h2>
                  <p className="text-muted mb-0">{stat.text}</p>
                </div>
              </Col>
            ))}
          </Row>
        </Container>
      </section>
      
      {/* Detection Forms Section */}
      <section ref={detectionRef} className="py-5">
        <Container>
          <div className="section-heading text-center mb-5">
            <h2>Choose Detection Type</h2>
            <p className="text-muted">
              Upload content to analyze and detect if it's AI-generated or authentic
            </p>
          </div>
          
          <Row className="justify-content-center">
            <Col lg={8}>
              <Tabs defaultActiveKey="video" className="mb-4">
                <Tab eventKey="video" title={<><i className="fas fa-video me-2"></i>Video</>}>
                  <div className="p-4 bg-light rounded shadow-sm">
                    <DetectionForm 
                      detectionType="Video Deepfake"
                      acceptedFileTypes={['video/mp4', 'video/quicktime', 'video/x-msvideo']}
                      onSubmit={handleDetectionSubmit}
                    />
                  </div>
                </Tab>
                
                <Tab eventKey="document" title={<><i className="fas fa-file-alt me-2"></i>Document</>}>
                  <div className="p-4 bg-light rounded shadow-sm">
                    <DetectionForm 
                      detectionType="Document Forgery"
                      acceptedFileTypes={['application/pdf', 'image/jpeg', 'image/png']}
                      onSubmit={handleDetectionSubmit}
                    />
                  </div>
                </Tab>
                
                <Tab eventKey="signature" title={<><i className="fas fa-signature me-2"></i>Signature</>}>
                  <div className="p-4 bg-light rounded shadow-sm">
                    <DetectionForm 
                      detectionType="Signature Forgery"
                      acceptedFileTypes={['image/jpeg', 'image/png', 'image/tiff']}
                      onSubmit={handleDetectionSubmit}
                    />
                  </div>
                </Tab>
              </Tabs>
            </Col>
          </Row>
        </Container>
      </section>
      
      {/* Results Section */}
      {results.length > 0 && (
        <section className="py-5 bg-light">
          <Container>
            <div className="section-heading mb-4">
              <h2>Detection Results</h2>
              <p className="text-muted">
                Results are listed in chronological order with the most recent at the top
              </p>
            </div>
            
            <div className="results-section">
              {results.map((result, index) => (
                <ResultPanel key={index} result={result} />
              ))}
            </div>
          </Container>
        </section>
      )}
      
      {/* Features Section */}
      <section className="py-5 bg-pattern">
        <Container>
          <div className="section-heading text-center mb-5">
            <h2>Why Choose COPixel</h2>
            <p className="text-muted">
              Our AI detection system offers unmatched capabilities to protect you from digital fraud
            </p>
          </div>
          
          <Row>
            {[
              {
                icon: 'fa-shield-alt',
                title: 'High Accuracy',
                text: 'Our algorithms achieve over 99.5% accuracy in detecting AI-generated content.'
              },
              {
                icon: 'fa-tachometer-alt',
                title: 'Fast Processing',
                text: 'Get results in seconds, not minutes, with our optimized detection engine.'
              },
              {
                icon: 'fa-eye',
                title: 'Comprehensive Detection',
                text: 'Detect deepfakes, document forgeries, and signature manipulations in one platform.'
              },
              {
                icon: 'fa-chart-line',
                title: 'Continuous Learning',
                text: 'Our AI models continuously improve to detect the latest forgery techniques.'
              },
              {
                icon: 'fa-lock',
                title: 'Secure & Private',
                text: 'Your uploads are encrypted and automatically deleted after analysis.'
              },
              {
                icon: 'fa-file-export',
                title: 'Detailed Reports',
                text: 'Get comprehensive reports with confidence scores and detailed analysis.'
              }
            ].map((feature, index) => (
              <Col lg={4} md={6} className="mb-4" key={index}>
                <div className="info-card p-4">
                  <div className="d-flex align-items-center mb-3">
                    <div className="icon-wrapper me-3">
                      <i className={`fas ${feature.icon}`}></i>
                    </div>
                    <h3 className="fw-bold m-0">{feature.title}</h3>
                  </div>
                  <p className="mb-0">{feature.text}</p>
                </div>
              </Col>
            ))}
          </Row>
        </Container>
      </section>
      
      {/* CTA Section */}
      <section className="py-5 hero-section">
        <Container className="text-center">
          <h2 className="display-5 fw-bold mb-4">Ready to detect AI-generated content?</h2>
          <p className="lead mb-5">
            Start using our advanced AI detection tools today and protect yourself from digital fraud
          </p>
          <Button 
            size="lg" 
            variant="light" 
            className="btn-lg px-5 py-3 rounded-pill fw-bold"
            onClick={scrollToDetection}
          >
            <i className="fas fa-shield-alt me-2"></i> Start Detection Now
          </Button>
        </Container>
      </section>
    </>
  );
}

export default HomePage; 