-- Run this first in Supabase SQL Editor to see your actual table names
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
