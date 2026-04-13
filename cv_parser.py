"""
CV Parser Module — Extract structured text from CV PDFs
Supports both English and Vietnamese CVs with bilingual section detection
"""

import pdfplumber
import re
import os


class CVParser:
    """Parse a CV from PDF file, extract structured text and metadata."""

    # Common section header patterns (English + Vietnamese bilingual)
    SECTION_PATTERNS = {
        'contact': r'(?i)(contact|liên\s*hệ|thông\s*tin\s*(cá\s*nhân|liên\s*hệ)|personal\s*info)',
        'summary': r'(?i)(summary|objective|giới\s*thiệu|mục\s*tiêu|tóm\s*tắt|profile|about\s*me)',
        'experience': r'(?i)(experience|kinh\s*nghiệm|work\s*history|lịch\s*sử\s*làm\s*việc|employment)',
        'education': r'(?i)(education|học\s*vấn|trình\s*độ|đào\s*tạo|academic)',
        'skills': r'(?i)(skill|kỹ\s*năng|technical|công\s*nghệ|tech\s*stack|competenc)',
        'certifications': r'(?i)(certif|chứng\s*chỉ|license|bằng\s*cấp)',
        'projects': r'(?i)(project|dự\s*án|portfolio)',
        'languages': r'(?i)(language|ngôn\s*ngữ|ngoại\s*ngữ|foreign)',
        'awards': r'(?i)(award|giải\s*thưởng|achievement|thành\s*tích)',
        'interests': r'(?i)(interest|sở\s*thích|hobbies|hobby)',
        'references': r'(?i)(reference|người\s*tham\s*chiếu|referees)',
    }

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.sections = {}
        self.metadata = {}

    def parse(self) -> dict:
        """Parse PDF and return structured data."""
        self._extract_text()
        self._detect_language()
        self._extract_sections()
        self._extract_metadata()
        return self.get_result()

    def _extract_text(self):
        """Extract all text content from PDF."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"File not found: {self.pdf_path}")

        text_parts = []
        with pdfplumber.open(self.pdf_path) as pdf:
            self.metadata['pages'] = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        self.raw_text = "\n".join(text_parts)
        if not self.raw_text.strip():
            raise ValueError("Cannot extract text from PDF. File may be a scanned image.")

    def _detect_language(self):
        """Detect the primary language of the CV."""
        vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
                                          self.raw_text.lower()))
        total_chars = len(self.raw_text)
        ratio = vietnamese_chars / max(total_chars, 1)
        self.metadata['language'] = 'vi' if ratio > 0.01 else 'en'
        self.metadata['vietnamese_char_ratio'] = round(ratio, 4)

    def _extract_sections(self):
        """Parse and split CV into logical sections."""
        lines = self.raw_text.split('\n')
        current_section = 'header'
        self.sections = {'header': []}

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if this line is a section header
            matched_section = None
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.search(pattern, line_stripped):
                    # Short line (<80 chars) matching pattern → section header
                    if len(line_stripped) < 80:
                        matched_section = section_name
                        break

            if matched_section:
                current_section = matched_section
                if current_section not in self.sections:
                    self.sections[current_section] = []
            else:
                if current_section not in self.sections:
                    self.sections[current_section] = []
                self.sections[current_section].append(line_stripped)

    def _extract_metadata(self):
        """Extract metadata from CV (email, phone, name, etc.)."""
        text = self.raw_text

        # Email
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if emails:
            self.metadata['email'] = emails[0]

        # Phone
        phones = re.findall(r'(?:\+84|0)\s*\d[\d\s.-]{7,13}', text)
        if phones:
            self.metadata['phone'] = phones[0].strip()

        # LinkedIn
        linkedin = re.findall(r'linkedin\.com/in/[\w-]+', text)
        if linkedin:
            self.metadata['linkedin'] = linkedin[0]

        # GitHub
        github = re.findall(r'github\.com/[\w-]+', text)
        if github:
            self.metadata['github'] = github[0]

        # Extract years of experience (supports "X năm" and "X years" patterns)
        years_vi = re.findall(r'(\d+)\s*(?:năm|years?)\s*(?:kinh\s*nghiệm|experience)', text, re.IGNORECASE)
        years_en = re.findall(r'(\d+)\+?\s*years?\s*(?:of\s+)?experience', text, re.IGNORECASE)
        years = years_vi + years_en
        if years:
            self.metadata['years_experience'] = max(int(y) for y in years)

    def get_result(self) -> dict:
        """Return parsed result as dictionary."""
        sections_text = {}
        for section, lines in self.sections.items():
            sections_text[section] = '\n'.join(lines)

        return {
            'raw_text': self.raw_text,
            'sections': sections_text,
            'metadata': self.metadata,
            'word_count': len(self.raw_text.split()),
        }


def parse_cv(pdf_path: str) -> dict:
    """Convenience function to parse a CV."""
    parser = CVParser(pdf_path)
    return parser.parse()


if __name__ == '__main__':
    # Test with sample CV
    import json
    cv_path = os.path.join(os.path.dirname(__file__), 'Mau_1', 'CV_NGUYEN_CONG_LAP_NHAN_AI_ENGINEER_2.pdf')
    result = parse_cv(cv_path)
    print(f"Language: {result['metadata'].get('language')}")
    print(f"Pages: {result['metadata'].get('pages')}")
    print(f"Word count: {result['word_count']}")
    print(f"\nSections found: {list(result['sections'].keys())}")
    print(f"\nMetadata: {json.dumps(result['metadata'], indent=2, ensure_ascii=False)}")
    print(f"\n--- Raw Text (first 500 chars) ---")
    print(result['raw_text'][:500])
