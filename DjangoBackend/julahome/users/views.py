import copy
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.response import Response
from .serializers import UserSerializer, AccountDataSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import BasicAuthentication


# Create your views here.

@api_view(["POST"])
def register_account(request):
    data = request.data

    serializer = UserSerializer(data=request.data)

    if serializer.is_valid(raise_exception=True):

        serializer.save()
        ret_data = copy.deepcopy(serializer.data)
        ret_data.pop("password")
        ret_data["registered"] = True

        return Response(ret_data, status=200)
    else:
        return Response({"invalid": "missing elements in data"}, status=400)


@api_view(["GET"])
@authentication_classes([BasicAuthentication])
@permission_classes([IsAuthenticated])
def get_account_details(request):

    user = str(request.user)

    model_data = get_user_model().objects.get(pk=user)
    print(model_data)

    serializer = AccountDataSerializer(model_data)

    return Response(serializer.data, status=200)
