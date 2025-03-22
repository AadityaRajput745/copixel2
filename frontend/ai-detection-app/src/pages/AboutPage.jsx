import { Container, Row, Col, Card } from 'react-bootstrap';

function AboutPage() {
  return (
    <>
      <section className="hero-section py-5">
        <Container>
          <Row className="justify-content-center text-center">
            <Col lg={8}>
              <h1 className="display-4 fw-bold">About COPixel</h1>
              <p className="lead">
                Leading the fight against AI-generated deepfakes and digital forgery with cutting-edge technology
              </p>
            </Col>
          </Row>
        </Container>
      </section>
      
      <section className="py-5 bg-pattern">
        <Container>
          <Row className="mb-5 align-items-center">
            <Col lg={6} className="mb-4 mb-lg-0">
              <div className="fade-in">
                <div className="badge badge-primary mb-3">Our Mission</div>
                <h2 className="fw-bold mb-4">Protecting Digital Integrity</h2>
                <p className="lead mb-4">
                  To protect digital integrity and trust by providing advanced tools for detecting AI-generated content.
                </p>
                <p className="mb-4">
                  As AI technology advances, the creation of convincing fake content becomes increasingly accessible.
                  Our mission is to stay one step ahead by developing detection systems that can identify these 
                  sophisticated forgeries and help maintain trust in digital content.
                </p>
                <div className="d-flex align-items-center mb-4">
                  <div className="icon-wrapper me-3">
                    <i className="fas fa-check"></i>
                  </div>
                  <p className="mb-0 fw-medium">We believe in a future where technology empowers people to make informed decisions about the content they consume.</p>
                </div>
                <div className="d-flex align-items-center">
                  <div className="icon-wrapper me-3">
                    <i className="fas fa-shield-alt"></i>
                  </div>
                  <p className="mb-0 fw-medium">We work tirelessly to ensure harmful misuse of AI can be quickly identified and addressed.</p>
                </div>
              </div>
            </Col>
            <Col lg={6}>
              <div className="position-relative slide-in">
                <img 
                  src="https://via.placeholder.com/600x400?text=Our+Mission"
                  alt="Our Mission" 
                  className="img-fluid rounded-lg shadow-lg"
                />
                <div className="position-absolute top-0 start-0 translate-middle-y p-4 bg-white shadow rounded-lg" style={{ left: '-20px' }}>
                  <div className="d-flex align-items-center">
                    <div className="icon-wrapper me-3 bg-success">
                      <i className="fas fa-check"></i>
                    </div>
                    <div>
                      <h5 className="fw-bold mb-0">Our Vision</h5>
                      <p className="mb-0 text-muted">A world free of digital deception</p>
                    </div>
                  </div>
                </div>
              </div>
            </Col>
          </Row>
          
          <div className="divider"></div>
          
          <Row className="mb-5 align-items-center">
            <Col lg={6} className="order-lg-2 mb-4 mb-lg-0">
              <div className="fade-in">
                <div className="badge badge-primary mb-3">Our Technology</div>
                <h2 className="fw-bold mb-4">Cutting-Edge AI Detection</h2>
                <p className="mb-4">
                  Our AI detection system uses a combination of advanced deep learning techniques to analyze content 
                  and identify signs of AI generation or manipulation.
                </p>
                <div className="mb-4">
                  <div className="d-flex align-items-center mb-3">
                    <div className="icon-wrapper me-3">
                      <i className="fas fa-user-shield"></i>
                    </div>
                    <h5 className="fw-bold mb-0">Facial Inconsistency Detection</h5>
                  </div>
                  <p className="ps-5 mb-0">Advanced algorithms that identify subtle inconsistencies in facial movements, blinks, and expressions that are telltale signs of deepfakes.</p>
                </div>
                <div className="mb-4">
                  <div className="d-flex align-items-center mb-3">
                    <div className="icon-wrapper me-3">
                      <i className="fas fa-file-alt"></i>
                    </div>
                    <h5 className="fw-bold mb-0">Document Forgery Detection</h5>
                  </div>
                  <p className="ps-5 mb-0">Multi-layered analysis of text, watermarks, and visual elements to identify manipulated or AI-generated documents.</p>
                </div>
                <div>
                  <div className="d-flex align-items-center mb-3">
                    <div className="icon-wrapper me-3">
                      <i className="fas fa-signature"></i>
                    </div>
                    <h5 className="fw-bold mb-0">Signature Verification</h5>
                  </div>
                  <p className="ps-5 mb-0">Precise stroke analysis and pattern recognition to authenticate signatures and detect forgeries.</p>
                </div>
              </div>
            </Col>
            <Col lg={6} className="order-lg-1">
              <div className="position-relative slide-in">
                <img 
                  src="https://via.placeholder.com/600x400?text=Our+Technology"
                  alt="Our Technology" 
                  className="img-fluid rounded-lg shadow-lg"
                />
                <div className="position-absolute top-100 end-0 translate-middle p-4 bg-white shadow rounded-lg" style={{ right: '-20px' }}>
                  <div className="d-flex align-items-center">
                    <div className="icon-wrapper me-3">
                      <i className="fas fa-brain"></i>
                    </div>
                    <div>
                      <h5 className="fw-bold mb-0">Continuous Learning</h5>
                      <p className="mb-0 text-muted">Our models adapt to new forgery techniques</p>
                    </div>
                  </div>
                </div>
              </div>
            </Col>
          </Row>
          
          <div className="divider"></div>
          
          <div className="text-center mb-5 fade-in">
            <div className="badge badge-primary mb-3">Our Team</div>
            <h2 className="fw-bold mb-4">Meet The Experts</h2>
            <p className="text-muted mx-auto" style={{ maxWidth: '700px' }}>
              We are a team of AI researchers, cybersecurity experts, and digital rights advocates 
              dedicated to fighting misinformation and digital forgery.
            </p>
          </div>
          
          <Row>
            {[
              {
                name: 'Dr. Sarah Chen',
                role: 'Chief AI Scientist',
                image: 'https://via.placeholder.com/300x300?text=Sarah+Chen',
                bio: 'Expert in deep learning and computer vision with 10+ years of experience in facial recognition systems.',
                social: {
                  linkedin: '#',
                  twitter: '#',
                  github: '#'
                }
              },
              {
                name: 'Michael Rodriguez',
                role: 'Lead Developer',
                image: 'https://via.placeholder.com/300x300?text=Michael+Rodriguez',
                bio: 'Full-stack developer specializing in secure systems and API development for AI applications.',
                social: {
                  linkedin: '#',
                  twitter: '#',
                  github: '#'
                }
              },
              {
                name: 'Aisha Patel',
                role: 'Digital Forensics Specialist',
                image: 'https://via.placeholder.com/300x300?text=Aisha+Patel',
                bio: 'Former law enforcement forensics expert with expertise in document verification and forgery detection.',
                social: {
                  linkedin: '#',
                  twitter: '#',
                  github: '#'
                }
              }
            ].map((member, index) => (
              <Col lg={4} md={6} className="mb-4" key={index}>
                <Card className="h-100 detector-card border-0 fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
                  <div className="position-relative">
                    <Card.Img variant="top" src={member.image} alt={member.name} className="team-img" />
                    <div className="team-social">
                      {Object.entries(member.social).map(([platform, link], idx) => (
                        <a href={link} key={idx} className="social-icon" target="_blank" rel="noopener noreferrer">
                          <i className={`fab fa-${platform}`}></i>
                        </a>
                      ))}
                    </div>
                  </div>
                  <Card.Body className="text-center">
                    <Card.Title as="h4" className="fw-bold">{member.name}</Card.Title>
                    <Card.Subtitle className="mb-3 text-primary fw-semibold">{member.role}</Card.Subtitle>
                    <Card.Text className="text-muted">{member.bio}</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>
        </Container>
      </section>
      
      <section className="py-5 bg-light">
        <Container>
          <div className="text-center mb-5 fade-in">
            <div className="badge badge-primary mb-3">Our Network</div>
            <h2 className="fw-bold mb-4">Trusted Partners</h2>
            <p className="text-muted mx-auto" style={{ maxWidth: '700px' }}>
              We work with leading academic institutions, media organizations, and technology companies 
              to advance the field of AI detection and verification.
            </p>
          </div>
          
          <Row className="justify-content-center align-items-center">
            {[1, 2, 3, 4, 5].map((num) => (
              <Col xs={6} md={4} lg={2} className="mb-4 px-4" key={num}>
                <div className="partner-logo fade-in" style={{ animationDelay: `${num * 0.1}s` }}>
                  <img 
                    src={`https://via.placeholder.com/150x80?text=Partner+${num}`}
                    alt={`Partner ${num}`}
                    className="img-fluid gray-img"
                  />
                </div>
              </Col>
            ))}
          </Row>
          
          <div className="text-center mt-5">
            <a href="/contact" className="btn btn-primary btn-lg px-5 py-3 rounded-pill">
              <i className="fas fa-handshake me-2"></i> Become a Partner
            </a>
          </div>
        </Container>
      </section>
      
      <style jsx="true">{`
        .team-img {
          transition: transform 0.5s ease;
          height: 300px;
          object-fit: cover;
        }
        
        .detector-card:hover .team-img {
          transform: scale(1.05);
        }
        
        .team-social {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          padding: 1rem;
          background: linear-gradient(to top, rgba(0,0,0,0.7), transparent);
          display: flex;
          justify-content: center;
          opacity: 0;
          transition: opacity 0.3s ease;
        }
        
        .detector-card:hover .team-social {
          opacity: 1;
        }
        
        .social-icon {
          background: white;
          color: var(--primary-color);
          width: 36px;
          height: 36px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 0.5rem;
          transition: transform 0.3s ease;
        }
        
        .social-icon:hover {
          transform: translateY(-5px);
        }
        
        .partner-logo {
          padding: 1.5rem;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: transform 0.3s ease;
        }
        
        .partner-logo:hover {
          transform: scale(1.1);
        }
        
        .gray-img {
          filter: grayscale(100%);
          opacity: 0.7;
          transition: all 0.3s ease;
        }
        
        .partner-logo:hover .gray-img {
          filter: grayscale(0%);
          opacity: 1;
        }
      `}</style>
    </>
  );
}

export default AboutPage; 