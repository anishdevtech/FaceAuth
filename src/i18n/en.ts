export default {
  // TabBar
  tab_verify: 'Verify',
  tab_enroll: 'Enroll',
  tab_manage: 'Manage',
  tab_logs: 'Logs',

  // Common
  cancel: 'CANCEL',
  retry: '↻',
  unknown_subject: 'UNKNOWN SUBJECT',

  // VerifyScreen
  verify_logged: 'LOGGED: {{name}}',
  verify_matched: 'MATCHED: {{name}} ({{confidence}}%)',
  verify_initializing: 'INITIALIZING MODELS...',
  verify_scanning: 'SCANNING FOR FACE...',
  initialization_note: 'This may take a few seconds on first load.',
  next_scan: 'NEXT SCAN',

  // EnrollScreen
  enroll_title: 'Enroll New Face',
  enroll_subtitle: 'Stand still · look at camera · press Capture',
  enroll_flip: 'FLIP',
  enroll_who: 'Who is this person?',
  enroll_enter_name: 'Enter their full name to register their face.',
  enroll_save: 'SAVE FACE',
  enroll_loading: 'Loading AI models…',

  // ManageScreen
  manage_title: 'Manage Profiles',
  manage_identities: '{{count}} IDENTITIES REGISTERED',
  manage_identity: '{{count}} IDENTITY REGISTERED',
  manage_clear_all: 'CLEAR ALL',
  manage_no_profiles: 'No profiles enrolled',
  manage_go_enroll: 'Go to the Enroll tab to add faces to the system.',
  manage_delete_face: 'Delete Face',
  manage_delete_face_sub: 'Are you sure you want to remove "{{name}}" from the system?',
  manage_clear_all_title: 'Clear All Faces',
  manage_clear_all_sub: 'This will permanently delete all enrolled faces. This action cannot be undone.',

  // LogsScreen
  logs_title: 'System Logs',
  logs_total_records: '{{count}} TOTAL RECORDS',
  logs_success: 'SUCCESS',
  logs_failed: 'FAILED',
  logs_event: 'EVENT:',
  logs_attendance: 'ATTENDANCE LOG',
  logs_enrollment: 'ENROLLMENT',
  logs_time: 'TIME:',
  logs_sync: 'SYNC:',
  logs_aws_uploaded: 'AWS UPLOADED',
  logs_pending: 'PENDING',
  logs_no_logs: 'NO LOGS FOUND',
  logs_scan_face: 'Scan a face in Verify mode to log attendance.',

  // Liveness
  liveness_left: 'Turn head left',
  liveness_right: 'Turn head right',
  liveness_up: 'Look up',
  liveness_down: 'Look down',
  liveness_blink: 'Blink eyes',
  liveness_smile: 'Smile',
};
