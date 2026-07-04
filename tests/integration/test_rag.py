# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from app.app_utils.guidelines_db import guidelines_db

def test_rag_offline_retrieval():
    """
    Tests that the guidelines vector database offline search fallback
    correctly retrieves matching guideline chunks using keyword matching.
    """
    # 1. Breast cancer query
    breast_chunks = guidelines_db.retrieve_guidelines("HER2 early breast cancer", cancer_type="breast")
    assert len(breast_chunks) > 0, "Should retrieve at least one breast cancer guideline chunk"
    assert any("HER2" in chunk for chunk in breast_chunks), "Retrieved chunk should contain HER2 information"
    
    # 2. Lung cancer query
    lung_chunks = guidelines_db.retrieve_guidelines("EGFR mutation metastatic", cancer_type="lung")
    assert len(lung_chunks) > 0, "Should retrieve at least one lung cancer guideline chunk"
    assert any("EGFR" in chunk for chunk in lung_chunks), "Retrieved chunk should contain EGFR mutation guidelines"
    
    print("Offline RAG retrieval test passed successfully!")
