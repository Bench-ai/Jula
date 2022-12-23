from django.contrib.auth import get_user_model
from rest_framework import authentication
from rest_framework import exceptions
from django.contrib.auth import authenticate
from .models import User


class EmailAuth(authentication.BaseAuthentication):
    def authenticate(self, request):
        email = request.META.get('HTTP_EMAIL')
        p_word = request.META.get('HTTP_PASSWORD')

        if not email or not p_word:  # no username passed in request headers
            return None  # authentication did not succeed

        try:
            user = get_user_model().objects.get(email__exact=email)  # get the user
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed('No such user')  # raise exception if user does not exist

        user = authenticate(username=user.uid, password=p_word)

        if not user:
            raise exceptions.AuthenticationFailed('Invalid email/password.')
        else:
            return user, None  # authentication successful
