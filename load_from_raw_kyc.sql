CREATE OR REPLACE PROCEDURE ssdx_eng.load_from_raw_kyc(
    p_load_individuals boolean DEFAULT true,
    p_load_corporates  boolean DEFAULT true
)
LANGUAGE plpgsql
AS $$
DECLARE
  v_ins bigint;
BEGIN
  ----------------------------------------------------------------------
  -- INDIVIDUALS
  ----------------------------------------------------------------------
  IF p_load_individuals THEN
    -- I1) Individuals → sdedba.ref_customer (no duplicates)
    INSERT INTO sdedba.ref_customer (
      customer_id, party_type_code, income_amount,
      registration_cou_id, registration_no, registration_date
    )
    SELECT DISTINCT
      nextval('SDEDBA.S_CUSTOMER'),
      7,
      NULLIF(i."Annual_Income",'')::numeric(16,2),
      3535,                         -- or NULL if you prefer for persons
      i."Legal_id",
      NULL::date
    FROM ssdx_tmp.individual_raw i
    WHERE NULLIF(i."Legal_id",'') IS NOT NULL
      AND NOT EXISTS (
        SELECT 1
        FROM sdedba.ref_customer rc
        WHERE rc.party_type_code = 7
          AND rc.registration_no = i."Legal_id"
      );

    -- I2) ONE misc_info row per customer (based on ref_customer)
    INSERT INTO sdedba.ref_customer_misc_info (
      customer_id,
      main_nationality_id,
      has_other_nationality,
      lgcy_customer_industry_code,
      lgcy_customer_sector_code,
      employment_category_code,
      expected_other_trx_amnt,
      expected_non_cash_trx_nbr,
      expected_cash_trx_nbr,
      situation_type_code
    )
    SELECT DISTINCT ON (rc.customer_id)
      rc.customer_id,
      1808,
      CASE
        WHEN NULLIF(i."Other_Nationality",'') IS NOT NULL
             AND UPPER(i."Other_Nationality") <> 'NONE'
        THEN TRUE ELSE FALSE END,
      /* text→numeric only if already numeric; else NULL */
      CASE WHEN i."Industry" ~ '^\d+(\.\d+)?$'          THEN i."Industry"::numeric          ELSE NULL END,
      CASE WHEN i."Sector" ~ '^\d+(\.\d+)?$'            THEN i."Sector"::numeric            ELSE NULL END,
      CASE WHEN i."Employment_Sector" ~ '^\d+(\.\d+)?$' THEN i."Employment_Sector"::numeric ELSE NULL END,
      NULLIF(i."Expected_Other_Transaction_Amount",'')::numeric(16,2),
      NULLIF(i."Expected_Non_Cash_Transactions",'')::numeric(16,2),
      NULLIF(i."Expected_Cash_Transactions",'')::numeric(16,2),
      CASE UPPER(TRIM(i."Professional_Status"))
        WHEN 'EMPLOYED' THEN 1
        WHEN 'SELF-EMPLOYED' THEN 2
        WHEN 'UNEMPLOYED' THEN 3
        WHEN 'STUDENT' THEN 4
        WHEN 'RETIRED' THEN 5
        ELSE NULL
      END::numeric
    FROM ssdx_tmp.individual_raw i
    JOIN sdedba.ref_customer rc
      ON rc.party_type_code = 7
     AND rc.registration_no = i."Legal_id"
    WHERE NOT EXISTS (
      SELECT 1 FROM sdedba.ref_customer_misc_info mi
      WHERE mi.customer_id = rc.customer_id
    )
    ORDER BY rc.customer_id,
             NULLIF(i."Employment_Joining_Date",'')::date DESC NULLS LAST,
             NULLIF(i."Monthly Income",'')::numeric(16,2) DESC NULLS LAST;

    -- I3) ONE JSON row per customer (based on ref_customer)
    INSERT INTO suitedba.cfg_customer_def (customer_def_id, customer_id, object_content,creation_date,created_by)
    SELECT DISTINCT ON (rc.customer_id)
      nextval('SUITEDBA.S_CUSTOMER_DEF'),
      rc.customer_id,
      jsonb_build_object(
        'LEGAL_PAPER_TYPE',     i."Legal Document Type",
        'LEGAL_PAPER_NUMBER',   i."Legal_id",
        'BRANCH',               i."Branch",
        'TITLE',                i."Title",
        'FIRST_NAME',           i."First_Name",
        'MIDDLE_NAME',          i."Middle_Name",
        'LAST_NAME',            i."Last_Name",
        'GENDER',               i."Gender",
        'DATE_OF_BIRTH',        NULLIF(i."Date Of Birth",''),
        'MOTHER_FULL_NAME',     i."Mother_Full_Name",
        'PLACE_OF_BIRTH',       i."Place_Of_Birth",
        'MOBILE_NUMBER',        i."Mobile_Number",
        'CITY',                 i."City",
        'AREA',                 i."Area",
        'STREET_NUMBER',        NULLIF(i."Street_Number",''),
        'STREET_NAME',          i."Stree_Name",
        'COUNTRY_OF_RESIDENCE', i."Country_of_Residence",
        'FLOOR',                NULLIF(i."Floor",''),
        'Building',             i."Building",
        'District',             i."District",
        'Zone',                 i."Zone",
        'Residence_Number',     i."Residence_Number",
        'Post_Code',            i."Post_Code",
        'Phone_Number',         i."Phone_Number",
        'Email',                i."Email",
        'Employment_Joining_Date', NULLIF(i."Employment_Joining_Date",''),
        'Monthly_Income',       NULLIF(i."Monthly Income",''),
        'Source_Of_Funds',      i."Source_Of_Funds",
        'Employment_Department',i."Employment_Department",
        'Education',            i."Education",
        'Office_Phone',         i."Office_Phone",
        'Do_you_have_any_additional_income', i."Do_You_Have_Any_Additional_Income?",
        'Wallet_Address',       i."Wallet_Address"
      ),current_date,7777
    FROM ssdx_tmp.individual_raw i
    JOIN sdedba.ref_customer rc
      ON rc.party_type_code = 7
     AND rc.registration_no = i."Legal_id"
    WHERE NOT EXISTS (
      SELECT 1 FROM suitedba.cfg_customer_def cd
      WHERE cd.customer_id = rc.customer_id
    )
    ORDER BY rc.customer_id,
             NULLIF(i."Employment_Joining_Date",'')::date DESC NULLS LAST,
             NULLIF(i."Monthly Income",'')::numeric(16,2) DESC NULLS LAST;
  END IF;

  ----------------------------------------------------------------------
  -- CORPORATES
  ----------------------------------------------------------------------
  IF p_load_corporates THEN
    -- C1) Corporates → sdedba.ref_customer (no duplicates)
    INSERT INTO sdedba.ref_customer (
      customer_id, party_type_code, income_amount,
      registration_cou_id, registration_no, registration_date
    )
    SELECT DISTINCT
      nextval('SDEDBA.S_CUSTOMER'),
      8,
      NULLIF(c."Annual_Income",'')::numeric(16,2),
      3535,  -- LB country ID
      COALESCE(NULLIF(c."Registration_No",''), NULLIF(c."Legal_id",'')),
      NULLIF(c."Registration_Date",'')::date
    FROM ssdx_tmp.corporate_raw c
    WHERE COALESCE(NULLIF(c."Registration_No",''), NULLIF(c."Legal_id",'')) IS NOT NULL
      AND NOT EXISTS (
        SELECT 1
        FROM sdedba.ref_customer rc
        WHERE rc.party_type_code = 8
          AND rc.registration_no = COALESCE(NULLIF(c."Registration_No",''), NULLIF(c."Legal_id",''))
      );

    -- C2) ONE misc_info row per corporate (lookup via inline maps; replace with real lookups if you have them)
    WITH
      ind_map(name, code) AS (
        VALUES ('Banking',1),('Telecommunications',2),('Retail',3),
               ('Construction',4),('Healthcare',5),('Education',6),
               ('Manufacturing',7),('IT Services',8),('Hospitality',9),
               ('Transportation',10),('Energy',11)
      ),
      sec_map(name, code) AS (
        VALUES ('Public',1),('Private',2),('NGO',3),('SME',4),('Startup',5),('Enterprise',6)
      ),
      emp_map(name, code) AS (
        VALUES ('Public',1),('Private',2),('NGO',3),('SME',4),('Startup',5),('Enterprise',6)
      ),
      sit_map(name, code) AS (
        VALUES ('Employed',1),('Self-Employed',2),('Unemployed',3),('Student',4),('Retired',5)
      )
    INSERT INTO sdedba.ref_customer_misc_info (
      customer_id,
      main_nationality_id,
      has_other_nationality,
      lgcy_customer_industry_code,
      lgcy_customer_sector_code,
      employment_category_code,
      expected_other_trx_amnt,
      expected_non_cash_trx_nbr,
      expected_cash_trx_nbr,
      situation_type_code
    )
    SELECT DISTINCT ON (rc.customer_id)
      rc.customer_id,
      NULL,
      FALSE,
      ind_map.code,
      sec_map.code,
      emp_map.code,
      NULLIF(c."Expected_Other_Transaction_Amount",'')::numeric(16,2),
      NULLIF(c."Expected_Non_Cash_Transactions",'')::numeric(16,2),
      NULLIF(c."Expected_Cash_Transactions",'')::numeric(16,2),
      sit_map.code
    FROM ssdx_tmp.corporate_raw c
    JOIN sdedba.ref_customer rc
      ON rc.party_type_code = 8
     AND rc.registration_no = COALESCE(NULLIF(c."Registration_No",''), NULLIF(c."Legal_id",''))
    LEFT JOIN ind_map ON ind_map.name = TRIM(c."Industry")
    LEFT JOIN sec_map ON sec_map.name = TRIM(c."Sector")
    LEFT JOIN emp_map ON emp_map.name = TRIM(c."Employment_Sector")
    LEFT JOIN sit_map ON sit_map.name = TRIM(c."Professional_Status")
    WHERE NOT EXISTS (
      SELECT 1 FROM sdedba.ref_customer_misc_info mi
      WHERE mi.customer_id = rc.customer_id
    )
    ORDER BY rc.customer_id,
             NULLIF(c."Employment_Joining_Date",'')::date DESC NULLS LAST,
             NULLIF(c."Monthly Income",'')::numeric(16,2) DESC NULLS LAST;

    -- C3) ONE JSON row per corporate
    INSERT INTO suitedba.cfg_customer_def (customer_def_id, customer_id, object_content,creation_date,created_by)
    SELECT DISTINCT ON (rc.customer_id)
      nextval('SUITEDBA.S_CUSTOMER_DEF'),
      rc.customer_id,
      jsonb_build_object(
        'LEGAL_PAPER_TYPE',                c."Legal Document Type",
        'LEGAL_PAPER_NUMBER',              c."Legal_id",
        'BRANCH',                          c."Branch",
        'REGISTRATION_EXPIRY_DATE',        NULLIF(c."Registration_Expiry_Date",''),
        'Business_License_Number',         c."Business_License_Number",
        'BUSINESS_LICENSE_ISSUE_DATE',     NULLIF(c."Business_License_Issue_Date",''),
        'BUSINESS_LICENSE_EXPIRY_DATE',    NULLIF(c."Business_License_Expiry_Date",''),
        'corporate_alias',                 c."Corporate_Alias",
        'MOBILE_NUMBER',                   c."Mobile_Number",
        'CITY',                            c."City",
        'AREA',                            c."Area",
        'STREET_NUMBER',                   NULLIF(c."Street_Number",''),
        'STREET_NAME',                     c."Stree_Name",
        'COUNTRY_OF_RESIDENCE',            c."Country_of_Residence",
        'FLOOR',                           NULLIF(c."Floor",''),
        'Building',                        c."Building",
        'District',                        c."District",
        'Zone',                            c."Zone",
        'Residence_Number',                c."Residence_Number",
        'Post_Code',                       c."Post_Code",
        'Phone_Number',                    c."Phone_Number",
        'Email',                           c."Email",
        'Employment_Joining_Date',         NULLIF(c."Employment_Joining_Date",''),
        'Monthly_Income',                  NULLIF(c."Monthly Income",''),
        'Source_Of_Funds',                 c."Source_Of_Funds",
        'Employment_Department',           c."Employment_Department",
        'Education',                       c."Education",
        'Office_Phone',                    c."Office_Phone",
        'Do_you_have_any_additional_income', c."Do_You_Have_Any_Additional_Income?"
      ),current_date,7777
    FROM ssdx_tmp.corporate_raw c
    JOIN sdedba.ref_customer rc
      ON rc.party_type_code = 8
     AND rc.registration_no = COALESCE(NULLIF(c."Registration_No",''), NULLIF(c."Legal_id",''))
    WHERE NOT EXISTS (
      SELECT 1 FROM suitedba.cfg_customer_def cd
      WHERE cd.customer_id = rc.customer_id
    )
    ORDER BY rc.customer_id,
             NULLIF(c."Employment_Joining_Date",'')::date DESC NULLS LAST,
             NULLIF(c."Monthly Income",'')::numeric(16,2) DESC NULLS LAST;
  END IF;

END;
$$;
