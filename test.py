import importlib.metadata
print(importlib.metadata.version("supabase"))

from supabase import create_client
from supabase_utils import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

data = supabase.table("feedback").select("*").limit(1).execute()
print(data)