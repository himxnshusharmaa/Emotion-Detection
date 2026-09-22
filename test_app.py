"""
Automated unit and integration tests for Emotion Classifier Flask app.
"""

import unittest
import json
from app import app, config, preprocess_text

class TestEmotionApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health(self):
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))
        self.assertEqual(len(data.get('classes', [])), 6)

    def test_examples(self):
        response = self.app.get('/api/examples')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertGreaterEqual(len(data.get('examples', [])), 6)

    def test_empty_predict(self):
        response = self.app.post('/api/predict', json={'text': '   '})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data.get('status'), 'error')

    def test_predict_joy(self):
        sample = "I am so extraordinarily happy and grateful for this incredible victory!"
        response = self.app.post('/api/predict', json={'text': sample})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['predicted_emotion'], 'joy')
        self.assertGreater(data['confidence'], 50.0)
        self.assertEqual(len(data['breakdown']), 6)
        
        # Verify probabilities sum to ~1.0
        prob_sum = sum(item['probability'] for item in data['breakdown'])
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_predict_sadness(self):
        sample = "i feel so hopeless and depressed today"
        response = self.app.post('/api/predict', json={'text': sample})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['predicted_emotion'], 'sadness')

    def test_predict_anger(self):
        sample = "I am absolutely furious and pissed off about this insult"
        response = self.app.post('/api/predict', json={'text': sample})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['predicted_emotion'], 'anger')

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("EmotionSense", html)
        self.assertIn("Detect Emotion", html)

    def test_preprocessing(self):
        raw = "Hello World! 123 😊 I am test."
        cleaned = preprocess_text(raw)
        # Should be lowercase, punctuation removed, digits removed, emojis removed, stopwords removed
        self.assertNotIn('123', cleaned)
        self.assertNotIn('!', cleaned)
        self.assertNotIn('😊', cleaned)
        self.assertEqual(cleaned, "hello world test")

if __name__ == '__main__':
    unittest.main()
