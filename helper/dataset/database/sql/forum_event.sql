
  SELECT
  AccountUserID as user_id,
  EventID as forum_id,
  b.EventType as forum_type,
  b.TimeStamp as timestamp,
  c.PostType,
  c.PostTitle,
  c.PostText

  FROM  ca_{platform}.Forum_Events as b
  LEFT JOIN ca_courseware.Forum_Info as c
  ON b.PostID = c.PostID
  AND b.DataPackageID = c.DataPackageID

  WHERE b.DataPackageID in ('{course}')
  LIMIT 2000;
