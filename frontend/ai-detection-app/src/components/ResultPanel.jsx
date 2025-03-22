import { Card, Row, Col, ProgressBar, Badge, Button } from 'react-bootstrap';
import { useState } from 'react';

function ResultPanel({ result }) {
  const [expanded, setExpanded] = useState(false);
  
  // Calculate confidence color based on score
  const getConfidenceColor = (score) => {
    if (score > 90) return 'success';
    if (score > 70) return 'info';
    if (score > 50) return 'warning';
    return 'danger';
  };
  
  const confidenceColor = getConfidenceColor(result.confidenceScore);
  
  // Format timestamp to readable format
  const formattedDate = new Date(result.timestamp).toLocaleString();
  
  // Decision badge based on result
  const decisionBadge = result.isAI ? (
    <Badge bg="danger" className="detection-badge ai-badge">
      <i className="fas fa-robot me-1"></i> AI-Generated
    </Badge>
  ) : (
    <Badge bg="success" className="detection-badge authentic-badge">
      <i className="fas fa-check-circle me-1"></i> Authentic
    </Badge>
  );
  
  return (
    <Card className="result-panel border-0 shadow-sm mb-4 rounded-lg fade-in">
      <Card.Body className="p-4">
        <Row className="align-items-center">
          <Col md={2} className="mb-3 mb-md-0">
            <div className="result-thumbnail">
              {result.thumbnailUrl ? (
                <img 
                  src={result.thumbnailUrl} 
                  alt={`Thumbnail for ${result.filename}`} 
                  className="img-fluid rounded" 
                />
              ) : (
                <div className="placeholder-thumbnail d-flex align-items-center justify-content-center text-white">
                  {result.detectionType === 'Video Deepfake' && <i className="fas fa-video fa-2x"></i>}
                  {result.detectionType === 'Document Forgery' && <i className="fas fa-file-alt fa-2x"></i>}
                  {result.detectionType === 'Signature Forgery' && <i className="fas fa-signature fa-2x"></i>}
                </div>
              )}
            </div>
          </Col>
          
          <Col md={7} className="mb-3 mb-md-0">
            <div className="d-flex align-items-center mb-2">
              {decisionBadge}
              <Badge bg="secondary" className="ms-2 detection-type-badge">
                {result.detectionType}
              </Badge>
            </div>
            
            <h4 className="mb-2 text-truncate">{result.filename}</h4>
            
            <div className="mb-3">
              <small className="text-muted">
                <i className="far fa-clock me-1"></i> {formattedDate}
              </small>
            </div>
            
            <div className="confidence-meter mb-2">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span>Confidence Score</span>
                <span className={`fw-bold text-${confidenceColor}`}>{result.confidenceScore}%</span>
              </div>
              <ProgressBar 
                variant={confidenceColor} 
                now={result.confidenceScore} 
                className="confidence-bar"
              />
            </div>
          </Col>
          
          <Col md={3} className="text-md-end">
            <div className="d-grid gap-2">
              <Button 
                variant="outline-primary" 
                className="rounded-pill btn-sm"
                onClick={() => setExpanded(!expanded)}
              >
                <i className={`fas fa-angle-${expanded ? 'up' : 'down'} me-1`}></i>
                {expanded ? 'Less Details' : 'More Details'}
              </Button>
              
              <Button 
                variant="outline-danger" 
                className="rounded-pill btn-sm"
                onClick={() => window.open('/report', '_blank')}
              >
                <i className="fas fa-flag me-1"></i> Report
              </Button>
            </div>
          </Col>
        </Row>
        
        {expanded && (
          <div className="mt-4 expanded-details animate__animated animate__fadeIn">
            <hr />
            <Row>
              <Col md={6} className="mb-3 mb-md-0">
                <h5 className="mb-3">Detection Details</h5>
                <ul className="detail-list">
                  {result.detectionFeatures && result.detectionFeatures.map((feature, index) => (
                    <li key={index} className={feature.detected ? 'text-danger' : 'text-success'}>
                      <i className={`fas fa-${feature.detected ? 'times' : 'check'} me-2`}></i>
                      {feature.name}: {feature.value}
                    </li>
                  ))}
                  
                  {!result.detectionFeatures && (
                    result.isAI ? (
                      <>
                        <li className="text-danger">
                          <i className="fas fa-times me-2"></i>
                          Inconsistent patterns detected
                        </li>
                        <li className="text-danger">
                          <i className="fas fa-times me-2"></i>
                          AI generation markers present
                        </li>
                      </>
                    ) : (
                      <>
                        <li className="text-success">
                          <i className="fas fa-check me-2"></i>
                          No inconsistencies found
                        </li>
                        <li className="text-success">
                          <i className="fas fa-check me-2"></i>
                          Authentic content verified
                        </li>
                      </>
                    )
                  )}
                </ul>
              </Col>
              
              <Col md={6}>
                <h5 className="mb-3">Recommendation</h5>
                <Card className={`border-${result.isAI ? 'danger' : 'success'} bg-light`}>
                  <Card.Body>
                    <div className="d-flex">
                      <div className="me-3">
                        <i className={`fas fa-${result.isAI ? 'exclamation-triangle text-danger' : 'check-circle text-success'} fa-2x`}></i>
                      </div>
                      <div>
                        {result.isAI ? (
                          <>
                            <p className="mb-2">This content appears to be AI-generated. We recommend:</p>
                            <ul className="mb-0">
                              <li>Verify with the purported source</li>
                              <li>Check for additional context</li>
                              <li>Look for other verification markers</li>
                              <li>Report if being used deceptively</li>
                            </ul>
                          </>
                        ) : (
                          <>
                            <p className="mb-2">This content appears to be authentic. However:</p>
                            <ul className="mb-0">
                              <li>No detection system is 100% accurate</li>
                              <li>Always verify critical content through multiple means</li>
                              <li>Be cautious of content that seems suspicious regardless of detection results</li>
                            </ul>
                          </>
                        )}
                      </div>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Card.Body>
      
      <style jsx>{`
        .result-panel {
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          overflow: hidden;
        }
        
        .result-panel:hover {
          transform: translateY(-5px);
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08) !important;
        }
        
        .result-thumbnail {
          width: 100%;
          aspect-ratio: 16/9;
          overflow: hidden;
          border-radius: 8px;
          background-color: #f0f0f0;
        }
        
        .placeholder-thumbnail {
          height: 100%;
          background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        }
        
        .detection-badge {
          padding: 0.5rem 0.75rem;
          border-radius: 20px;
          font-weight: 600;
          font-size: 0.8rem;
          box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        }
        
        .ai-badge {
          animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
          0% {
            box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4);
          }
          70% {
            box-shadow: 0 0 0 10px rgba(220, 53, 69, 0);
          }
          100% {
            box-shadow: 0 0 0 0 rgba(220, 53, 69, 0);
          }
        }
        
        .detection-type-badge {
          font-size: 0.7rem;
          padding: 0.4rem 0.6rem;
          background-color: #6c757d;
          font-weight: 400;
        }
        
        .confidence-bar {
          height: 8px;
          border-radius: 4px;
        }
        
        .detail-list {
          list-style: none;
          padding-left: 0;
        }
        
        .detail-list li {
          margin-bottom: 0.5rem;
          padding: 0.5rem;
          border-radius: 5px;
          background-color: rgba(0, 0, 0, 0.03);
        }
      `}</style>
    </Card>
  );
}

export default ResultPanel; 