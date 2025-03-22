import { Card } from 'react-bootstrap';
import { Link } from 'react-router-dom';

function DetectorCard({ title, description, icon, link, accuracy }) {
  return (
    <Card className="detector-card h-100 border-0 shadow-sm">
      <div className="detector-icon">
        {icon}
        {accuracy && <span className="accuracy-badge">{accuracy}</span>}
      </div>
      <Card.Body className="p-4">
        <Card.Title as="h3" className="fw-bold mb-3">{title}</Card.Title>
        <Card.Text className="mb-4">{description}</Card.Text>
        <Link to={link} className="btn btn-primary rounded-pill">
          <i className="fas fa-shield-alt me-2"></i> Try Detection
        </Link>
      </Card.Body>
      
      <style jsx>{`
        .detector-card {
          border-radius: 12px;
          overflow: hidden;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          position: relative;
        }
        
        .detector-card:hover {
          transform: translateY(-10px);
          box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1) !important;
        }
        
        .detector-icon {
          width: 60px;
          height: 60px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
          position: absolute;
          top: -20px;
          left: 20px;
          z-index: 2;
          box-shadow: 0 8px 20px rgba(var(--primary-rgb), 0.3);
          transition: transform 0.3s ease;
        }
        
        .detector-card:hover .detector-icon {
          transform: scale(1.1) rotate(10deg);
        }
        
        .accuracy-badge {
          position: absolute;
          bottom: -10px;
          right: -10px;
          background-color: #fff;
          color: var(--primary-color);
          border-radius: 20px;
          font-size: 12px;
          font-weight: bold;
          padding: 2px 8px;
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
      `}</style>
    </Card>
  );
}

export default DetectorCard; 