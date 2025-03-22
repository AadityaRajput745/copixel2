import { useState, useEffect } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { Navbar as BootstrapNavbar, Nav, Container } from 'react-bootstrap';
import copixel_logo from '../assets/copixel_logo.jpg';

function Navbar({ toggleDarkMode, isDarkMode }) {
  const [scrolled, setScrolled] = useState(false);

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setScrolled(true);
      } else {
        setScrolled(true);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <BootstrapNavbar 
      variant="dark" 
      expand="lg" 
      className={`navbar-dark fixed-top ${scrolled ? 'scrolled' : ''}`}
    >
      <Container>
        <BootstrapNavbar.Brand as={Link} to="/">
          <img 
            src="/src/assets/copixel_logo.jpg" 
            width="35" 
            height="35" 
            className="d-inline-block align-top me-2 rounded-circle" 
            alt="COPixel Logo"
          />
          <span className="d-none d-sm-inline">COPixel</span>
        </BootstrapNavbar.Brand>
        <BootstrapNavbar.Toggle aria-controls="basic-navbar-nav" />
        <BootstrapNavbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link as={NavLink} to="/" end className="nav-link-hover">
              <i className="fas fa-home me-1"></i> Home
            </Nav.Link>
            <Nav.Link as={NavLink} to="/about" className="nav-link-hover">
              <i className="fas fa-info-circle me-1"></i> About
            </Nav.Link>
            <Nav.Link as={NavLink} to="/contact" className="nav-link-hover">
              <i className="fas fa-envelope me-1"></i> Contact
            </Nav.Link>
            <Nav.Link 
              as="button" 
              onClick={toggleDarkMode} 
              className="ms-2 nav-link-hover btn-link"
              aria-label="Toggle dark mode"
            >
              {isDarkMode ? (
                <i className="fas fa-sun"></i>
              ) : (
                <i className="fas fa-moon"></i>
              )}
            </Nav.Link>
          </Nav>
        </BootstrapNavbar.Collapse>
      </Container>
    </BootstrapNavbar>
  );
}

export default Navbar; 